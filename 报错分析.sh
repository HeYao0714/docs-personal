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






--- MiMo-V2.5-Pro-FP4-DFlash/hle_Biology/Medicine ---
  predictions:  1150547 B, 3 行 [完整]
  reviews:      2287638 B, 3 行 [完整]
    #0: target=['B'] | answer= ❌
    #1: target=['False'] | answer=true ❌
    #2: target=['(1,4,5), (1,3,4,5,6)'] | answer= ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Chemistry ---
  predictions:   410885 B, 3 行 [完整]
  reviews:       673766 B, 3 行 [完整]
    #0: target=['1.776 * 10^-3'] | answer=1.8 * 10^-3 ❌
    #1: target=['1.86'] | answer=$d \approx 1.86 \times 10^{28}$ ❌
    #2: target=['Al, Re2Al13; Al, ReAl12; Al, Re2Al9'] | answer=re_a, real12; al_a, al12; al_b, al11; al_c, al10 ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Computer Science/AI ---
  predictions:   971456 B, 3 行 [完整]
  reviews:      1793435 B, 3 行 [完整]
    #0: target=['Katie kicked the knotted kite string, knowing it would take skill to unknot the tangled mess.'] | answer="katie kicked the knotted kite string, knowing it would take skill to unknot the tangled mess." ❌
    #1: target=['E'] | answer= ❌
    #2: target=['C'] | answer= ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Engineering ---
  predictions:    95833 B, 3 行 [完整]
  reviews:        95084 B, 3 行 [完整]
    #0: target=['F'] | answer=a ❌
    #1: target=['10'] | answer=30 ❌
    #2: target=['B'] | answer=a ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Humanities/Social Science ---
  predictions:    27035 B, 3 行 [完整]
  reviews:        31179 B, 3 行 [完整]
    #0: target=['D'] | answer=d. weak non-sadism ❌
    #1: target=['Sale Law'] | answer= ❌
    #2: target=['Yes'] | answer=no, the account is both descriptive and normative ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Math ---
  predictions:   648949 B, 3 行 [完整]
  reviews:       922790 B, 3 行 [完整]
    #0: target=['Z+Z+Z+Z+Z'] | answer=$\mathbb{z}^5$ ❌
    #1: target=['18'] | answer= ❌
    #2: target=['$1 + 3x + 6x^2 + 8x^3 + 6x^4 + 3x^5 + x^6$'] | answer=$x^6 + 3x^5 + 6x^4 + 8x^3 + 6x^2 + 2x + 1$ ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Other ---
  predictions:   162880 B, 3 行 [完整]
  reviews:       103438 B, 3 行 [完整]
    #0: target=['yeyo'] | answer=** concatenation of c1, c2, c4, c5 (all lowercase): ❌
    #1: target=['C'] | answer=d ❌
    #2: target=['Shuriken'] | answer=stiletto ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Physics ---
  predictions:   829354 B, 3 行 [完整]
  reviews:      1490503 B, 3 行 [完整]
    #0: target=['\\(-((d - 2k)^2) + d\\)'] | answer={answer} ❌
    #1: target=['3'] | answer= ❌
    #2: target=['\\begin{pmatrix}8&9\\\\9&8\\end{pmatrix}'] | answer=$\begin{pmatrix}-2&-1\\-1&-2\end{pmatrix}$ ❌

================================================================================
汇总
================================================================================
  完整 predictions: 8/8
  完整 reviews:     8/8
  可评分学科:       8/8
  可评分样本:       24
  正确:             0/24 = 0.00%
  阈值 0.33:  FAIL

可评分的学科: MiMo-V2.5-Pro-FP4-DFlash/hle_Biology/Medicine, MiMo-V2.5-Pro-FP4-DFlash/hle_Chemistry, MiMo-V2.5-Pro-FP4-DFlash/hle_Computer Science/AI, MiMo-V2.5-Pro-FP4-DFlash/hle_Engineering, MiMo-V2.5-Pro-FP4-DFlash/hle_Humanities/Social Science, MiMo-V2.5-Pro-FP4-DFlash/hle_Math, MiMo-V2.5-Pro-FP4-DFlash/hle_Other, MiMo-V2.5-Pro-FP4-DFlash/hle_Physics




--- MiMo-V2.5-Pro-FP4-DFlash/hle_Biology/Medicine ---
  predictions:  1150547 B, 3 行 [完整]
  reviews:      2287638 B, 3 行 [完整]
    #1: target=['False'] | answer=true ❌
    #2: target=['(1,4,5), (1,3,4,5,6)'] | answer= ❌
    #0: target=['B'] | answer= ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Chemistry ---
  predictions:   410885 B, 3 行 [完整]
  reviews:       673766 B, 3 行 [完整]
    #0: target=['1.776 * 10^-3'] | answer=1.8 * 10^-3 ❌
    #2: target=['Al, Re2Al13; Al, ReAl12; Al, Re2Al9'] | answer=re_a, real12; al_a, al11re; al_b, al10re; al_c, al9re ❌
    #1: target=['1.86'] | answer=$d \approx 1.86 \times 10^{28}$ ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Computer Science/AI ---
  predictions:   971456 B, 3 行 [完整]
  reviews:      1793435 B, 3 行 [完整]
    #0: target=['Katie kicked the knotted kite string, knowing it would take skill to unknot the tangled mess.'] | answer="katie kicked the knotted kite string, knowing it would take skill to unknot the ❌
    #1: target=['E'] | answer= ❌
    #2: target=['C'] | answer= ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Engineering ---
  predictions:    95833 B, 3 行 [完整]
  reviews:        95084 B, 3 行 [完整]
    #0: target=['F'] | answer=a ❌
    #1: target=['10'] | answer=30 ❌
    #2: target=['B'] | answer=a ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Humanities/Social Science ---
  predictions:    27035 B, 3 行 [完整]
  reviews:        31179 B, 3 行 [完整]
    #0: target=['D'] | answer=d. weak non-sadism ❌
    #1: target=['Sale Law'] | answer= ❌
    #2: target=['Yes'] | answer=no, the account is both descriptive and normative ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Math ---
  predictions:   648949 B, 3 行 [完整]
  reviews:       922790 B, 3 行 [完整]
    #0: target=['Z+Z+Z+Z+Z'] | answer=$\mathbb{z}^5$ ❌
    #2: target=['$1 + 3x + 6x^2 + 8x^3 + 6x^4 + 3x^5 + x^6$'] | answer=$x^6 + 3x^5 + 6x^4 + 8x^3 + 6x^2 + 2x + 1$ ❌
    #1: target=['18'] | answer= ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Other ---
  predictions:   162880 B, 3 行 [完整]
  reviews:       103438 B, 3 行 [完整]
    #0: target=['yeyo'] | answer=ytyo ❌
    #2: target=['Shuriken'] | answer=stiletto ❌
    #1: target=['C'] | answer=d ❌

--- MiMo-V2.5-Pro-FP4-DFlash/hle_Physics ---
  predictions:   829354 B, 3 行 [完整]
  reviews:      1490503 B, 3 行 [完整]
    #0: target=['\\(-((d - 2k)^2) + d\\)'] | answer=$d - (d-2k)^2$ ❌
    #1: target=['3'] | answer= ❌
    #2: target=['\\begin{pmatrix}8&9\\\\9&8\\end{pmatrix}'] | answer=$\begin{pmatrix}-2&-1\\-1&-2\end{pmatrix}$ ❌

