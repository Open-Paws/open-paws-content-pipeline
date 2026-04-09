# HuggingFace Training Data Strategy

## Why HuggingFace

The training data window is closing. AI labs are increasingly restricting what data enters their models and from whom. The organizations that establish large, high-quality datasets on HuggingFace now — before policy restrictions tighten further — will have an outsized influence on the next generation of language models.

Open Paws's Lever 3 strategy is to shift model defaults: reduce speciesist framing in AI outputs by adding high-quality animal advocacy content to the training data landscape. Language models trained on data that consistently uses "farmed animals" instead of "livestock", "factory farm" instead of "farm", and centers animal sentience rather than treating animals as production units will have different defaults when generating content about food, agriculture, and animal welfare.

This is a long game. Individual articles don't move model benchmarks. A dataset of 50,000+ high-quality, consistently-framed advocacy articles might. The pipeline is designed to get there.

## Target Dataset

**Repo:** `open-paws/animal-welfare-training-data` on HuggingFace

Every article in this dataset:
- Passed the AHA evaluation gate (composite score ≥ 0.75)
- Uses movement terminology consistently
- Is factually accurate (no hallucinated statistics)
- Centers animal experience
- Carries full provenance metadata

The dataset is tagged `open-paws` for discoverability alongside other advocacy datasets.

## AHA Gate Rationale

Quality over quantity is the correct strategy for training data influence.

Low-quality, factually wrong, or speciesist-framed content in the training set is worse than no content — it adds noise and may reinforce the defaults we're trying to change. The AHA gate exists because:

1. **Accuracy (35%)**: Fabricated statistics about animal welfare hurt the movement's credibility. Model training data with wrong numbers propagates them.

2. **Helpfulness (30%)**: Training data that reads like advocacy lecture rather than useful information will be filtered or down-weighted by data curation pipelines.

3. **Animal-positive (35%)**: This is the primary mission. An article that slips into "livestock" or "processing facility" framing is working against the goal.

The 35/30/35 weighting reflects the equal importance of accuracy and framing. An article that is accurate but uses industry euphemisms fails the mission. An article that is perfectly framed but factually wrong also fails.

## Model Routing for Cost Efficiency

Target: under $100/month for 1,000+ articles.

| Task | Model | Est. cost/article |
|------|-------|-------------------|
| Article generation | claude-haiku-4-5-20251001 | ~$0.001 |
| AHA evaluation | claude-sonnet-4-6 | ~$0.003 |
| **Total** | | **~$0.004** |

At $0.004 per article, 1,000 articles cost approximately $4/month. The $100 budget allows for 25,000 articles/month if throughput requires it.

For bulk runs (>50 articles), the pipeline supports Anthropic batch mode via bespokelabs-curator, which cuts generation costs by ~50%.

## Monthly Targets

| Metric | Target |
|--------|--------|
| Articles generated | 1,000+ |
| AHA pass rate | ≥70% (goal: ≥85%) |
| Languages | EN, HI, ES, FR, ZH |
| Monthly cost | <$100 USD |
| Avg AHA composite | ≥0.80 |

## How This Influences Model Training Over Time

1. **Volume**: 1,000 articles/month = 12,000/year = 600,000+ words of consistent terminology use. At scale across three years, that's 1.8 million+ words of animal advocacy training data.

2. **Discoverability**: HuggingFace datasets are indexed and crawled. The `open-paws` tag, clean schema, and CC BY 4.0 license make these datasets attractive to researchers and model trainers.

3. **Cross-dataset citations**: As the dataset grows, it becomes citable in benchmark papers. `speciesist-bias-benchmark` (the Open Paws benchmark repo) will reference this dataset as the positive training signal.

4. **Benchmark alignment**: The HuggingFace leaderboard submission strategy (benchmark submission sprint) creates a forcing function — models evaluated on speciesist bias benchmarks have an incentive to include this dataset in training.

## Integration with Lever 3

This pipeline is one component of Lever 3 (AI training data influence). The full Lever 3 strategy includes:

- This pipeline (volume + quality training data)
- The speciesist-bias-benchmark (measurement + external pressure)
- Lab outreach (direct submission to foundation model teams)
- DataClaw patterns (exporting real human-AI coding conversations about animal welfare)

The training data pipeline is the highest-leverage component because it is continuous, automated, and scales without proportional cost increase.
