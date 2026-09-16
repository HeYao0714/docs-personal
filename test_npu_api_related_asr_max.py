def test_max_buffer(self):
        # Load a local WAV file and convert it to a PCM16 byte stream.
        full_pcm = self.load_wav_to_pcm16(WAV_PATH, TARGET_SR)
        samples_per_chunk = int(CHUNK_SEC * TARGET_SR)
        bytes_per_sample = 2
        chunk_bytes_len = samples_per_chunk * bytes_per_sample

        ws = websocket.create_connection(WS_URL, timeout=10)
        error = None
        transcript_text = None
        try:
            # Handle the initial `session.created` event and add exception handling.
            try:
                init_raw = ws.recv()
                init_msg = json.loads(init_raw)
                logging.warning("Initial service message:", init_msg)
            except WebSocketTimeoutException:
                self.fail(
                    "Service unresponsive: Initial `session.created` event not received within 10 seconds of connecting to WebSocket."
                )

            # Update session configuration
            session_update_msg = json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": TARGET_SR},
                                "transcription": {"model": "qwen3-asr"},
                            }
                        },
                    },
                }
            )
            ws.send(session_update_msg)
            # wait session.updated
            ws.settimeout(10)
            session_ok = False
            for _ in range(20):
                try:
                    evt_raw = ws.recv()
                    evt = json.loads(evt_raw)
                    if evt["type"] == "session.updated":
                        session_ok = True
                        break
                    if evt["type"] == "error":
                        error = evt
                        break
                except (WebSocketTimeoutException, WebSocketConnectionClosedException):
                    continue
            self.assertTrue(
                session_ok,
                "Failed to wait for the session.updated configuration timeout.",
            )

            offset = 0
            total_sec = 0.0
            ws.settimeout(1.0)

            # Cyclically shard and transmit the complete audio.
            while offset < len(full_pcm):
                end = offset + chunk_bytes_len
                chunk = full_pcm[offset:end]
                offset = end
                send_msg = json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
                ws.send(send_msg)

                total_sec += CHUNK_SEC
                logging.warning(f"Total duration of sent audio: {total_sec:.1f}s")

                # Catching Server-Side Errors
                try:
                    evt_raw = ws.recv()
                    evt = json.loads(evt_raw)
                    if evt["type"] == "error":
                        error = evt
                        logging.warning("Received a server-side error:", error)
                        self.assertIn(
                            "Accumulated audio exceeded", error["error"]["message"]
                        )
                        break
                except (WebSocketTimeoutException, WebSocketConnectionClosedException):
                    continue

            # No errors occurred; sending a commit triggers full inference.
            if not error:
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                ws.settimeout(30)
                max_wait_loop = 60
                finish_flag = False
                for _ in range(max_wait_loop):
                    try:
                        resp_raw = ws.recv()
                        resp = json.loads(resp_raw)
                        if (
                            resp["type"]
                            == "conversation.item.input_audio_transcription.completed"
                        ):
                            transcript_text = resp["transcript"]
                            logging.warning("Final version:", transcript_text)
                            finish_flag = True
                            break
                        if resp["type"] == "error":
                            error = resp
                            self.assertIn(
                                "Accumulated audio exceeded",
                                error["error"]["message"],
                            )
                            finish_flag = True
                            break
                    except (
                        WebSocketTimeoutException,
                        WebSocketConnectionClosedException,
                    ):
                        logging.warning("Timed out waiting for transcription event.")
                        continue
                self.assertTrue(
                    finish_flag,
                    f"Transcription completion event not received within 30 seconds.",
                )
                if transcript_text is not None:
                    self.assertGreater(
                        len(transcript_text.strip()),
                        0,
                        "Transcription result is empty.",
                    )

        finally:
            ws.close()


def create_active_session(self, idx, result_list, lock):
        res = {"idx": idx, "ok": False, "msg": ""}
        ws = None
        try:
            ws = websocket.create_connection(WS_URL, timeout=3)

            # Receive the first message
            first_raw = ws.recv()
            first_msg = json.loads(first_raw)

            if first_msg.get("type") == "error":
                err = first_msg.get("error", {})
                res["msg"] = (
                    f"Service rejected | code={err.get('code')} | {err.get('message')}"
                )

            elif first_msg.get("type") != "session.created":
                res["msg"] = (
                    f"Session creation event not received; actual type is: {first_msg.get('type')}"
                )

            else:
                time.sleep(2)
                res["ok"] = True
                res["msg"] = "Active session successfully created."
        except Exception as e:
            res["msg"] = f"Connection error: {str(e)}"
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

        # Thread-safe writing of results
        with lock:
            result_list.append(res)
