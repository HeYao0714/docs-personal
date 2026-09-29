def _update_weights(
        self, named_tensors: List[Tuple[str, torch.Tensor]]
    ) -> requests.Response:
        requests.post(f"{self.url}/begin_weight_update", json={}, timeout=120)
        resp = requests.post(
            f"{self.url}/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
                ],
                "flush_cache": True,
            },
            timeout=120,
        )
        requests.post(f"{self.url}/end_weight_update", json={}, timeout=120)
        return resp


def test_e_checksum_returns_ranks_with_hashes(self):
        """checksum action must yield a ranks list with hex hashes per rank."""
        resp = self._post("checksum")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertIn("ranks", body)
        ranks = body["ranks"]
        self.assertIsInstance(ranks, list)
        self.assertGreaterEqual(len(ranks), 1)

        first = ranks[0]
        self.assertIn("checksums", first)
        self.assertIn("parallelism_info", first)

        infos = first["parallelism_info"]
        # one entry per runner; without speculative decoding that is the target
        self.assertEqual([info["role"] for info in infos], ["target"])
        info = infos[0]
        for key in (
            "tp_rank",
            "tp_size",
            "dp_rank",
            "dp_size",
            "pp_rank",
            "pp_size",
            "rank",
            "size",
        ):
            self.assertIn(key, info)

        checksums = first["checksums"]
        self.assertGreater(len(checksums), 0)
        for name, h in checksums.items():
            self.assertIsInstance(h, str)
            self.assertEqual(len(h), 16, f"unexpected hash length for {name!r}: {h!r}")
            int(h, 16)
