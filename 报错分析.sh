Traceback (most recent call last):
  File "/tmp/evalscope_run_MiMo-V2.5-Pro-FP4-DFlash_hle.py", line 40, in <module>
    run_task(task_cfg=task_cfg)
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/test_env_evalscope/lib/python3.12/site-packages/evalscope/run.py", line 33, in run_task
    return run_single_task(task_cfg)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/test_env_evalscope/lib/python3.12/site-packages/evalscope/run.py", line 48, in run_single_task
    result = evaluate_model(task_cfg, outputs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/test_env_evalscope/lib/python3.12/site-packages/evalscope/run.py", line 213, in evaluate_model
    res_dict = evaluator.eval()
               ^^^^^^^^^^^^^^^^
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/test_env_evalscope/lib/python3.12/site-packages/evalscope/evaluator/evaluator.py", line 226, in eval
    report = self.get_report(agg_score_dict, execution_summary)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/test_env_evalscope/lib/python3.12/site-packages/evalscope/evaluator/evaluator.py", line 553, in get_report
    report = self.benchmark.generate_report(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/test_env_evalscope/lib/python3.12/site-packages/evalscope/api/benchmark/adapters/default_data_adapter.py", line 908, in generate_report
    report = self._on_generate_report(scores, model_name=model_name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/test_env_evalscope/lib/python3.12/site-packages/evalscope/api/benchmark/adapters/default_data_adapter.py", line 890, in _on_generate_report
    return ReportGenerator.generate_report(score_dict=scores, model_name=model_name, data_adapter=self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/test_env_evalscope/lib/python3.12/site-packages/evalscope/report/generator.py", line 136, in generate_report
    raise ValueError(
ValueError: No scores were collected for dataset "hle". Please check that samples are not all filtered out and that the aggregation step produces results.
2026-09-22 21:43:26,946 - INFO - run_evalscope finished: pid=259954 returncode=1
2026-09-22 21:43:26,946 - INFO - process group pgid=259954 already gone
2026-09-22 21:43:26,947 - ERROR - Command failed with return code: 1
2026-09-22 21:43:26,947 - ERROR - Error executing command: Command 'test_env_evalscope/bin/python /tmp/evalscope_run_MiMo-V2.5-Pro-FP4-DFlash_hle.py' returned non-zero exit status 1.
2026-09-22 21:43:26,947 - INFO - process group pgid=259954 already gone







root@localhost:/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro# python3 analyze_hle_results.py /root/.cache/tests/output/accuracy/20260922/test_npu_mimo_v2_5_pro_w4a8_8p_a5_hle/20260922_125841
================================================================================
日志目录: /root/.cache/tests/output/accuracy/20260922/test_npu_mimo_v2_5_pro_w4a8_8p_a5_hle/20260922_125841
发现 8 个学科
================================================================================

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Biology/Medicine ---
  predictions:  1150547 B, 3 行 [完整]
  reviews:      2287638 B, 3 行 [完整]
Traceback (most recent call last):
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/analyze_hle_results.py", line 230, in <module>
    main()
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/analyze_hle_results.py", line 196, in main
    answers = extract_answer_from_prediction(pred_files[subject])
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro/analyze_hle_results.py", line 93, in extract_answer_from_prediction
    text_blocks = [b.get("text", "") for b in content if b.get("type") == "text"]
                                                         ^^^^^
AttributeError: 'str' object has no attribute 'get'
root@localhost:/home/h30085291/sglang/test/registered/npu/accuracy/mimo_v2_5_pro# 

