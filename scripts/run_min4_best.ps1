$ErrorActionPreference = "Stop"

$Config = "config/cstamoerec_all_beauty_min4.yaml"
$SaveDir = "checkpoints/cstamoerec_min4_best"
$Checkpoint = "$SaveDir/best_cstamoerec.pt"
$CandidateCheckpoint = "$SaveDir/best_candidate_reranker.pt"
$Ranker = "$SaveDir/learned_source_ranker.json"

$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

Write-Host "== Prepare min4-best data =="
python scripts/prepare_amazon2023.py --config $Config --device cuda --text-batch-size 256

Write-Host "== Train modal-aware LightGCN graphs =="
python scripts/train_lightgcn.py --config $Config --epochs 20 --batch-size 4096 --similarity-topk 50 --similarity-batch-size 384 --device cuda

Write-Host "== Train CS-TAMoERec =="
python scripts/train_cstamoerec.py --config $Config --device cuda

Write-Host "== Train candidate reranker =="
python scripts/train_candidate_reranker.py --config $Config --checkpoint-in $Checkpoint --checkpoint-out $CandidateCheckpoint --per-source-k 500 --max-candidates 1000 --epochs 8 --lr 0.0001 --device cuda

Write-Host "== Train learned source ranker =="
python scripts/tune_source_ranker.py --config $Config --per-source-k 500 --max-candidates 1000 --epochs 250 --output $Ranker --device cuda

Write-Host "== Evaluate sampled-negative baselines =="
python scripts/evaluate_traditional_baselines.py --config $Config --checkpoint $Checkpoint --num-negatives 99 --device cuda

Write-Host "== Evaluate candidate recall =="
python scripts/evaluate_candidates.py --config $Config --split test --per-source-k 500 --max-candidates 1000 --topk 50 100 200 500 1000 --checkpoint $Checkpoint --include-model-candidates --device cuda

Write-Host "== Evaluate true two-stage modes =="
python scripts/rerank_candidates.py --config $Config --checkpoint $Checkpoint --mode hybrid --split test --per-source-k 500 --max-candidates 1000 --include-model-candidates --device cuda
python scripts/rerank_candidates.py --config $Config --checkpoint $Checkpoint --mode learned_source --source-ranker $Ranker --split test --per-source-k 500 --max-candidates 1000 --include-model-candidates --device cuda
python scripts/rerank_candidates.py --config $Config --checkpoint $CandidateCheckpoint --mode adaptive --split test --per-source-k 500 --max-candidates 1000 --include-model-candidates --device cuda

Write-Host "== Explainability and demo artifacts =="
python scripts/evaluate_perturbation.py --config $Config --checkpoint $Checkpoint --split test --device cuda
python scripts/evaluate_counterfactual.py --config $Config --checkpoint $Checkpoint --split test --limit-users 200 --device cuda
python scripts/summarize_experiments.py --config $Config
python scripts/export_demo_cache.py --config $Config --checkpoint $Checkpoint --mode learned_source --source-ranker $Ranker --only-hits --num-users 50 --per-source-k 500 --max-candidates 1000 --topk 10 --include-model-candidates --device cuda

Write-Host "Done. Report: $SaveDir/experiment_report.md"
