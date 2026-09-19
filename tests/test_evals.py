from llm_obs import EvalSample, HeuristicJudge, LLMJudge, evaluate_dataset

CONTEXT = "Paris is the capital of France. France is in Europe."
QUESTION = "What is the capital of France?"


def _score_map(scores):
    return {s.name: s for s in scores}


def test_grounded_answer_scores_high():
    sample = EvalSample(input=QUESTION, output="The capital of France is Paris.", context=CONTEXT)
    scores = _score_map(HeuristicJudge(threshold=0.5).score(sample))
    assert scores["groundedness"].passed
    assert scores["relevance"].passed


def test_hallucinated_answer_fails_groundedness():
    sample = EvalSample(input=QUESTION, output="Bananas grow in tropical climates.", context=CONTEXT)
    scores = _score_map(HeuristicJudge(threshold=0.5).score(sample))
    assert not scores["groundedness"].passed


def test_evaluate_dataset_aggregates():
    samples = [
        EvalSample(QUESTION, "The capital of France is Paris.", CONTEXT),
        EvalSample(QUESTION, "Bananas grow in tropical climates.", CONTEXT),
    ]
    summary = evaluate_dataset(samples, HeuristicJudge(threshold=0.5))
    assert summary["groundedness"].n == 2
    assert summary["groundedness"].pass_rate == 0.5   # one grounded, one not


def test_llm_judge_parses_scores():
    judge = LLMJudge(complete=lambda prompt: '{"groundedness": 0.9, "relevance": 0.8}', threshold=0.7)
    scores = _score_map(judge.score(EvalSample(QUESTION, "Paris.", CONTEXT)))
    assert scores["groundedness"].value == 0.9
    assert scores["groundedness"].passed and scores["relevance"].passed


def test_llm_judge_handles_non_json():
    judge = LLMJudge(complete=lambda prompt: "sorry, I cannot")
    scores = judge.score(EvalSample(QUESTION, "Paris.", CONTEXT))
    assert len(scores) == 1 and scores[0].name == "parse_error" and not scores[0].passed
