"""Post-meeting tasks — aggregation, webhooks, hooks.

Post-meeting aggregation, webhooks, and hooks.
Same logic, same webhook payloads.
"""

import logging
import os

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Meeting
from .database import async_session_local
from .outbound_events import claim_outbound_event, event_key, mark_outbound_event
from .webhook_delivery import deliver_with_result, build_envelope

from .config import TRANSCRIPTION_COLLECTOR_URL, POST_MEETING_HOOKS
from .webhooks import send_completion_webhook

logger = logging.getLogger("meeting_api.post_meeting")


# v0.10.5 Pack H — aggregation_failure_class taxonomy.
#
# JSONB discriminator (NO PG migration; meetings.data is JSONB and
# meetings.status is String(50) — both already accept new keys/values
# without schema changes). Three values:
#   - "transient_infra"      — tx-gateway 5xx; retry-eligible via Pack H.4 sweep
#   - "permanent_infra"      — 7-day retry budget exhausted; terminal; alerts critical
#   - "user_meaningful"      — bot crash, validation error, etc; terminal as `status=failed`
#
# Single-write-path discipline: every write goes through
# set_aggregation_failure_class() helper. Pack H's registry check
# AGGREGATION_FAILURE_CLASS_VIA_TYPED_HELPER asserts no other call site
# touches data['aggregation_failure_class'] directly.
class AggregationFailureClass:
    TRANSIENT_INFRA = "transient_infra"
    PERMANENT_INFRA = "permanent_infra"
    USER_MEANINGFUL = "user_meaningful"


def set_aggregation_failure_class(meeting: Meeting, cls: str) -> None:
    """Canonical single-write-path for data.aggregation_failure_class.

    Updates meeting.data dict in place + flags it modified for SQLAlchemy.
    Caller commits.
    """
    valid = {
        AggregationFailureClass.TRANSIENT_INFRA,
        AggregationFailureClass.PERMANENT_INFRA,
        AggregationFailureClass.USER_MEANINGFUL,
    }
    if cls not in valid:
        raise ValueError(f"Invalid aggregation_failure_class: {cls!r}; must be one of {valid}")
    data = dict(meeting.data or {})
    data["aggregation_failure_class"] = cls
    from datetime import datetime
    data["aggregation_last_retry_at"] = datetime.utcnow().isoformat()
    data["aggregation_retry_count"] = (data.get("aggregation_retry_count") or 0) + 1
    meeting.data = data
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(meeting, "data")


def clear_aggregation_failure_class(meeting: Meeting) -> None:
    """Clear failure_class on success — also via the canonical write path."""
    if not meeting.data:
        return
    data = dict(meeting.data)
    if "aggregation_failure_class" in data:
        del data["aggregation_failure_class"]
        meeting.data = data
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(meeting, "data")


async def aggregate_transcription(meeting: Meeting, db: AsyncSession):
    """Fetch transcription segments and aggregate into meeting.data.

    v0.10.5 Pack H — distinguishes transient infra failures (5xx, network
    error) from permanent failures (4xx auth/validation) from success.
    Pre-Pack-H: ALL non-200 returned silently; meeting marked `failed`
    via callers' default-to-completed/failed paths regardless of cause.
    Real-world incident 2026-04-23: tx-gateway pod restart during
    aggregate → 23 consecutive meetings marked `failed` from a transient
    flap.

    Now:
      - 5xx OR network error → set aggregation_failure_class='transient_infra'
        — retry-eligible via Pack H.4 sweep in sweeps.py (24 retries × exp
        backoff, 7-day budget). Caller stays in non-terminal state until
        either retry succeeds or budget exhausts.
      - 4xx → set aggregation_failure_class='permanent_infra' (e.g. auth
        misconfig); terminal; alerts critical. Operator action required.
      - 200 + segments → clear failure_class; aggregate normally.

    Returns True on terminal success, False on transient (caller can choose
    to surface aggregation_failed event vs leave for sweep retry).
    """
    meeting_id = meeting.id
    try:
        collector_url = f"{TRANSCRIPTION_COLLECTOR_URL}/internal/transcripts/{meeting_id}"
        internal_secret = os.getenv("INTERNAL_API_SECRET", "")
        headers = {"X-Internal-Secret": internal_secret} if internal_secret else {}
        async with httpx.AsyncClient() as client:
            response = await client.get(collector_url, timeout=30.0, headers=headers)

        # v0.10.5 Pack H — distinguish 5xx (transient) from 4xx (permanent).
        if 500 <= response.status_code < 600:
            logger.warning(
                f"Pack H: tx-gateway returned {response.status_code} for meeting {meeting_id} "
                f"— transient infra, retrying via sweep"
            )
            set_aggregation_failure_class(meeting, AggregationFailureClass.TRANSIENT_INFRA)
            await db.commit()
            return False
        if response.status_code != 200:
            # 4xx — permanent. Auth misconfig, malformed request, etc.
            logger.error(
                f"Pack H: tx-gateway returned {response.status_code} for meeting {meeting_id} "
                f"— permanent infra failure (operator action required)"
            )
            set_aggregation_failure_class(meeting, AggregationFailureClass.PERMANENT_INFRA)
            await db.commit()
            return False

        segments = response.json()
        if not segments:
            # Empty result is legitimate (zero-segment meeting); clear any
            # prior failure_class to indicate aggregation completed cleanly.
            clear_aggregation_failure_class(meeting)
            await db.commit()
            return True

        unique_speakers = set()
        unique_languages = set()
        for seg in segments:
            speaker = seg.get("speaker")
            language = seg.get("language")
            if speaker and speaker.strip():
                unique_speakers.add(speaker.strip())
            if language and language.strip():
                unique_languages.add(language.strip())

        existing_data = meeting.data or {}
        changed = False
        if "participants" not in existing_data and unique_speakers:
            existing_data["participants"] = sorted(unique_speakers)
            changed = True
        if "languages" not in existing_data and unique_languages:
            existing_data["languages"] = sorted(unique_languages)
            changed = True

        if changed:
            meeting.data = existing_data
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(meeting, "data")
            logger.info(f"Aggregated transcription data for meeting {meeting_id}")

        # Success — clear any prior transient failure_class.
        clear_aggregation_failure_class(meeting)
        await db.commit()
        return True

    except httpx.RequestError as exc:
        # Network error (DNS, TLS, timeout) — same class as 5xx; retry-eligible.
        logger.warning(
            f"Pack H: tx-gateway request error for meeting {meeting_id}: "
            f"{type(exc).__name__}: {exc!r} — transient infra, retrying via sweep"
        )
        set_aggregation_failure_class(meeting, AggregationFailureClass.TRANSIENT_INFRA)
        try:
            await db.commit()
        except Exception:
            pass
        return False
    except Exception as e:
        # Unknown error — log loudly + don't mark transient (don't retry into a code bug).
        logger.error(
            f"Pack H: aggregation failed for meeting {meeting_id}: "
            f"{type(e).__name__}: {e!r}",
            exc_info=True,
        )
        return False


async def fire_post_meeting_hooks(meeting: Meeting, db: AsyncSession) -> None:
    """Fire POST_MEETING_HOOKS to configured internal services (billing, analytics, etc.).

    v0.10.6.1 #330 — internal billing/usage hooks must be duplicate-safe
    and retryable without adding a database column. Each configured
    destination gets one ``meeting.data.outbound_events`` ledger entry under a
    ``SELECT ... FOR UPDATE`` row lock. The ledger claim commits before HTTP
    delivery, so concurrent callers either see a pending/queued/delivered event
    and skip, or the retry sweep can recover the narrow crash-after-claim
    window.
    """
    if not POST_MEETING_HOOKS:
        return

    if not meeting.start_time or not meeting.end_time:
        return

    # Resolve real email from users table — billing hooks need it to meter usage
    try:
        from admin_models.models import User
        user = (await db.execute(select(User).where(User.id == meeting.user_id))).scalars().first()
        if not user or not user.email:
            logger.error(f"Cannot resolve email for user {meeting.user_id} — skipping billing hook")
            return
        user_email = user.email
    except Exception as e:
        logger.error(f"DB error resolving email for user {meeting.user_id} — skipping billing hook: {e}")
        return

    duration_seconds = (meeting.end_time - meeting.start_time).total_seconds()
    meeting_data = meeting.data or {}

    event_data = {
        "meeting": {
            "id": meeting.id,
            "user_id": meeting.user_id,
            "user_email": user_email,
            "platform": meeting.platform,
            "status": meeting.status,
            "duration_seconds": duration_seconds,
            "start_time": meeting.start_time.isoformat(),
            "end_time": meeting.end_time.isoformat(),
            "created_at": meeting.created_at.isoformat() if meeting.created_at else None,
            "transcription_enabled": meeting_data.get("transcribe_enabled", False),
        },
    }

    for hook_url in POST_MEETING_HOOKS:
        key = event_key("post_meeting_hooks", "meeting.completed", meeting.id, hook_url)
        payload = build_envelope("meeting.completed", event_data, event_id=key)
        key, ledger_event, should_deliver = await claim_outbound_event(
            db,
            meeting_id=meeting.id,
            channel="post_meeting_hooks",
            event_type="meeting.completed",
            destination=hook_url,
            payload=payload,
        )
        if not should_deliver:
            logger.info(
                "fire_post_meeting_hooks: meeting %s hook %s already %s; skipping",
                meeting.id,
                key,
                ledger_event.get("status"),
            )
            continue

        result = await deliver_with_result(
            url=hook_url,
            payload=payload,
            timeout=10.0,
            label=f"post-meeting-hook meeting={meeting.id}",
            metadata={"meeting_id": meeting.id, "outbound_event_key": key},
        )
        await mark_outbound_event(
            db,
            meeting_id=meeting.id,
            key=key,
            status=result.status,
            attempts=int(ledger_event.get("attempts") or 0) + 1,
            error=result.error,
            status_code=result.response.status_code if result.response is not None else None,
        )


async def finalize_in_progress_recordings(meeting: Meeting, db: AsyncSession) -> int:
    """Mark all IN_PROGRESS recordings as COMPLETED + flip media_files[*].is_final=true.

    v0.10.5 (post-prod-telemetry 2026-04-30) — Bug B: pre-fix, recordings whose
    finalizer chunk never reached the server (bot was killed before it could send
    the empty-body is_final=true chunk) stayed IN_PROGRESS forever, with all
    media_files entries showing is_final=false. Consumers polling for is_final
    couldn't tell when the recording is truly done.

    Now: at post-meeting time (after meeting is in terminal state), any rec
    payload still IN_PROGRESS gets flipped to COMPLETED, and every media_files
    entry's is_final flag flipped to true. The actual chunk files in MinIO are
    already there; this is purely the metadata reconciliation.

    Returns count of recordings that were finalized here (0 if everything was
    already finalized via the canonical chunk-finalizer path).
    """
    from sqlalchemy.orm import attributes
    from .schemas import RecordingStatus
    from datetime import datetime as _dt

    if not meeting or not meeting.data:
        return 0
    recordings_list = list((meeting.data or {}).get("recordings") or [])
    if not recordings_list:
        return 0

    finalized_count = 0
    changed = False
    for idx, rec in enumerate(recordings_list):
        if not isinstance(rec, dict):
            continue
        # Only finalize recordings that haven't been completed via the
        # canonical chunk-finalizer path. Already-completed recordings stay
        # untouched.
        if rec.get("status") == RecordingStatus.COMPLETED.value:
            continue
        rec_payload = dict(rec)
        rec_payload["status"] = RecordingStatus.COMPLETED.value
        rec_payload["completed_at"] = rec_payload.get("completed_at") or _dt.utcnow().isoformat()
        # Flip is_final on every media_files entry so consumers see the
        # recording as done.
        media_files = list(rec_payload.get("media_files") or [])
        any_changed = False
        for mf in media_files:
            if not isinstance(mf, dict):
                continue
            # #311 — Single-writer policy for master_path.
            # If recording_finalizer has already written a master path on
            # this media_files entry, treat it as the canonical owner and
            # only observe — do NOT overwrite is_final / finalized_at /
            # finalized_by. The race window: recording_finalizer uploads
            # the master to storage and updates JSONB storage_path; if
            # post_meeting fires between those two writes (or right after
            # storage_path lands but before is_final flips), the
            # pre-#311 code would stomp finalized_by="post_meeting_reconciler"
            # over recording_finalizer's mark. Now: presence of a master
            # storage_path is the signal that recording_finalizer owns
            # this entry; we observe and skip.
            sp = (mf.get("storage_path") or "")
            if sp.endswith("/audio/master.webm") or sp.endswith("/audio/master.wav"):
                # Observed: recording_finalizer is in flight or done. Don't write.
                continue
            if not mf.get("is_final"):
                mf["is_final"] = True
                mf["finalized_at"] = _dt.utcnow().isoformat()
                mf["finalized_by"] = "post_meeting_reconciler"
                any_changed = True
        rec_payload["media_files"] = media_files
        recordings_list[idx] = rec_payload
        finalized_count += 1
        changed = changed or any_changed or True

    if changed:
        meeting_data_dict = dict(meeting.data or {})
        meeting_data_dict["recordings"] = recordings_list
        meeting.data = meeting_data_dict
        attributes.flag_modified(meeting, "data")
        logger.info(
            "[Bug-B-Fix] post_meeting_reconciler finalized recordings for meeting %s: count=%s",
            meeting.id, finalized_count,
        )
    return finalized_count


async def process_batch_transcription(meeting: Meeting, db: AsyncSession):
    import os, tempfile, wave, httpx, asyncio
    from .storage import create_storage_client
    from .models import Transcription
    from .meetings import _map_speakers_to_segments
    from sqlalchemy.future import select

    if not meeting.data or not meeting.data.get("transcribe_enabled", True):
        return

    # Check if transcripts already exist
    stmt = select(Transcription).where(Transcription.meeting_id == meeting.id)
    result = await db.execute(stmt)
    if result.scalars().first():
        logger.info(f"Transcripts already exist for meeting {meeting.id}, skipping batch processing.")
        return

    recordings = meeting.data.get("recordings", [])
    audio_path = None
    for rec in recordings:
        for mf in rec.get("media_files", []):
            if mf.get("type") == "audio" and (mf.get("storage_path", "").endswith("master.wav") or mf.get("storage_path", "").endswith("master.webm")):
                audio_path = mf["storage_path"]
                break
        if audio_path: break

    if not audio_path:
        logger.warning(f"No master audio found for meeting {meeting.id}, skipping batch transcription.")
        return

        groq_key = os.getenv("TRANSCRIPTION_SERVICE_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not groq_key:
        logger.error("No API key found for batch transcription.")
        return

    storage = create_storage_client()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        master_local_raw = os.path.join(tmpdir, "master_raw" + os.path.splitext(audio_path)[1])
        master_local = os.path.join(tmpdir, "master.wav")
        await asyncio.to_thread(storage.download_file_to_path, audio_path, master_local_raw)

        # Convert to WAV if needed
        import subprocess
        if master_local_raw.endswith(".webm"):
            try:
                subprocess.run(["ffmpeg", "-y", "-i", master_local_raw, "-ar", "16000", "-ac", "1", master_local], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.error(f"Failed to convert webm to wav for meeting {meeting.id}: {e}")
                return
        else:
            import shutil
            shutil.copy(master_local_raw, master_local)

        chunk_files = []
        with wave.open(master_local, 'rb') as w:
            framerate = w.getframerate()
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            frames_total = w.getnframes()
            
            # 10 minutes = 600 seconds
            frames_per_chunk = framerate * 600 
            
            frames_read = 0
            chunk_idx = 0
            while frames_read < frames_total:
                chunk_frames = w.readframes(frames_per_chunk)
                if not chunk_frames: break
                chunk_file = os.path.join(tmpdir, f"chunk_{chunk_idx}.wav")
                with wave.open(chunk_file, 'wb') as cw:
                    cw.setnchannels(nchannels)
                    cw.setsampwidth(sampwidth)
                    cw.setframerate(framerate)
                    cw.writeframes(chunk_frames)
                chunk_files.append((chunk_file, chunk_idx * 600))
                frames_read += frames_per_chunk
                chunk_idx += 1

        transcription_url = os.getenv("TRANSCRIPTION_BASE_URL", "https://api.groq.com/openai/v1/audio/transcriptions")
        transcription_model = os.getenv("TRANSCRIPTION_MODEL", "whisper-large-v3")
        
        all_words = []
        async with httpx.AsyncClient(timeout=180.0) as client:
            for c_file, offset_sec in chunk_files:
                with open(c_file, 'rb') as f:
                    files = {"file": ("audio.wav", f, "audio/wav")}
                    data = {"model": transcription_model, "response_format": "verbose_json", "language": "en"}
                    r = await client.post(
                        transcription_url,
                        headers={"Authorization": f"Bearer {groq_key}"},
                        files=files, data=data
                    )
                    if r.status_code != 200:
                        logger.error(f"Groq API error: {r.text}")
                        continue
                        
                    resp_json = r.json()
                    
                    if "words" in resp_json:
                        for w in resp_json["words"]:
                            all_words.append({"start": w["start"] + offset_sec, "end": w["end"] + offset_sec, "word": w["word"]})
                    elif "segments" in resp_json:
                        for s in resp_json["segments"]:
                            if "words" in s:
                                for w in s["words"]:
                                    all_words.append({"start": w["start"] + offset_sec, "end": w["end"] + offset_sec, "word": w["word"]})

        if not all_words: return

        stt_segments = []
        current_segment = {"speaker": "Speaker", "text": [], "start_time": all_words[0]["start"], "end_time": all_words[0]["end"]}
        
        for i, w in enumerate(all_words):
            word_text = w["word"].strip()
            if not word_text: continue
                
            current_segment["text"].append(word_text)
            current_segment["end_time"] = w["end"]
            
            is_end_of_sentence = word_text[-1] in ".!?"
            pause_duration = 0
            if i + 1 < len(all_words):
                pause_duration = all_words[i+1]["start"] - w["end"]
                
            if pause_duration > 1.5 or is_end_of_sentence or len(current_segment["text"]) > 20:
                current_segment["text"] = " ".join(current_segment["text"])
                stt_segments.append(current_segment)
                if i + 1 < len(all_words):
                    current_segment = {"speaker": "Speaker", "text": [], "start_time": all_words[i+1]["start"], "end_time": all_words[i+1]["end"]}

        if not stt_segments: return

        speaker_events = meeting.data.get("speaker_events", [])
        if speaker_events:
            for s in stt_segments:
                s["start_time_ms"] = int(s["start_time"] * 1000)
            stt_segments = _map_speakers_to_segments(speaker_events, stt_segments)

        for s in stt_segments:
            new_t = Transcription(
                meeting_id=meeting.id,
                speaker=s.get("speaker", "Speaker"),
                text=s["text"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                is_final=True,
                language="en"
            )
            db.add(new_t)
            
        await db.commit()
        logger.info(f"Batch transcription completed for meeting {meeting.id} with {len(stt_segments)} segments")



async def generate_ai_summary(meeting: Meeting, db: AsyncSession):
    from .models import Transcription
    from sqlalchemy.orm.attributes import flag_modified
    import os
    import httpx
    
    if not meeting.data or not meeting.data.get("transcribe_enabled", True):
        return

    # Skip if notes already exist and it's not empty
    existing_notes = meeting.data.get("notes", "").strip()
    if existing_notes:
        return

    groq_key = os.getenv("TRANSCRIPTION_SERVICE_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not groq_key:
        logger.warning("No API key found for AI summary generation.")
        return

    try:
        stmt = select(Transcription).where(Transcription.meeting_id == meeting.id).order_by(Transcription.start_time)
        result = await db.execute(stmt)
        segments = result.scalars().all()
        
        if not segments:
            return
            
        transcript_text = "\n".join([f"{seg.speaker}: {seg.text}" for seg in segments])
        
        if len(transcript_text.strip()) < 20:
            return

        prompt = f"Please generate a concise, professional summary of the following meeting transcript. Highlight key decisions, action items, and main discussion points. Be concise.\n\nTranscript:\n{transcript_text}"
        
        llm_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
        llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                llm_url,
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": llm_model,
                    "messages": [
                        {"role": "system", "content": "You are an AI meeting assistant that creates highly accurate and concise summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.3
                },
                timeout=60.0
            )
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"].strip()
            
            data = dict(meeting.data)
            data["notes"] = f"**AI Summary:**\n\n{summary}"
            meeting.data = data
            flag_modified(meeting, "data")
            logger.info(f"AI Summary generated for meeting {meeting.id}")
            
    except Exception as e:
        logger.error(f"Failed to generate AI summary for meeting {meeting.id}: {e}")


async def run_all_tasks(meeting_id: int):
    """Run all post-meeting tasks for a given meeting_id.

    Uses short-lived DB sessions to avoid holding connections during HTTP calls.
    """
    logger.info(f"Starting post-meeting tasks for meeting {meeting_id}")

    # Task 0 (v0.10.5 Bug B fix): finalize any IN_PROGRESS recordings whose
    # finalizer chunk never made it. Runs FIRST so downstream tasks (webhook
    # delivery, hooks) see the canonical "completed" state.
    try:
        async with async_session_local() as db:
            meeting = await db.get(Meeting, meeting_id)
            if meeting:
                count = await finalize_in_progress_recordings(meeting, db)
                if count > 0:
                    await db.commit()
    except Exception as e:
        logger.error(f"Recording finalization failed for meeting {meeting_id}: {e}", exc_info=True)

    # Task 1.1: Batch Transcription via Groq
    try:
        async with async_session_local() as db:
            meeting = await db.get(Meeting, meeting_id)
            if meeting:
                await process_batch_transcription(meeting, db)
    except Exception as e:
        logger.error(f"Batch transcription failed for meeting {meeting_id}: {e}", exc_info=True)

    # Task 1.2: Aggregate transcription chunks into a single document (for UI)
    try:
        async with async_session_local() as db:
            meeting = await db.get(Meeting, meeting_id)
            if meeting:
                await aggregate_transcription(meeting, db)
                await db.commit()
    except Exception as e:
        logger.error(f"Transcription aggregation failed for meeting {meeting_id}: {e}", exc_info=True)

    # Task 1.5: Generate AI Summary (makes HTTP call to Groq)
    try:
        async with async_session_local() as db:
            meeting = await db.get(Meeting, meeting_id)
            if meeting:
                await generate_ai_summary(meeting, db)
                await db.commit()
    except Exception as e:
        logger.error(f"AI summary generation failed for meeting {meeting_id}: {e}", exc_info=True)

    # Task 2: Send completion webhook to user (makes HTTP call to user's endpoint)
    try:
        async with async_session_local() as db:
            meeting = await db.get(Meeting, meeting_id)
            if meeting:
                await send_completion_webhook(meeting, db)
                await db.commit()
    except Exception as e:
        logger.error(f"Completion webhook failed for meeting {meeting_id}: {e}", exc_info=True)

    # Task 3: Fire internal post-meeting hooks (makes HTTP calls to hook URLs)
    try:
        async with async_session_local() as db:
            meeting = await db.get(Meeting, meeting_id)
            if meeting:
                await fire_post_meeting_hooks(meeting, db)
                await db.commit()
    except Exception as e:
        logger.error(f"Post-meeting hooks failed for meeting {meeting_id}: {e}", exc_info=True)

    logger.info(f"Post-meeting tasks completed for meeting {meeting_id}")


async def run_status_webhook_task(meeting_id: int, status_change_info: dict = None):
    """Run status webhook — short-lived DB session, HTTP call outside session."""
    from .webhooks import send_status_webhook

    try:
        async with async_session_local() as db:
            meeting = await db.get(Meeting, meeting_id)
            if not meeting:
                logger.error(f"Meeting {meeting_id} not found for status webhook")
                return
            await send_status_webhook(meeting, db, status_change_info)
            await db.commit()
    except Exception as e:
        logger.error(f"Error in status webhook for meeting {meeting_id}: {e}", exc_info=True)
