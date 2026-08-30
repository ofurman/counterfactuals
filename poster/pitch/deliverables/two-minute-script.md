# CEL: two-minute poster pitch

242 spoken words. Rehearse at approximately 121 words per minute; the timestamps are pacing targets, not an automatic timer. The same script is embedded in the single slide's speaker notes.

## 0:00-0:20

*Start with the problem; look at the audience.*

Imagine a model declines your loan application. A counterfactual explanation asks: what would need to change for the model to approve it? But which explanation method should we use? That question is difficult to answer when every study uses a different experimental setup.

## 0:20-0:45

*Introduce CEL and point to the four scope tiles.*

We introduce CEL: a benchmark and open-source library for counterfactual explanations. Our main contribution is a controlled evaluation protocol. We standardize datasets, preprocessing, predictive models, feature constraints, and metric definitions, so comparisons are less affected by experimental choices. The aim is to compare methods on common ground.

## 0:45-1:10

*Read across the four tiles: datasets, methods, backbones, metrics.*

The four tiles summarize the scope: eighteen datasets spanning classification and regression; fourteen methods across local, global, and group-wise explanations; two predictive backbones per task; and nine reported classification metrics. These cover success, change size, plausibility, and runtime, giving us a multidimensional picture rather than a single score.

## 1:10-1:40

*Point down the three result rows, then to the rightmost plausibility column.*

Adult Census illustrates trade-offs in all three paradigms. Locally, CADEX makes small changes but has lower validity. Globally, AReS often fails, while GLOBE-CE achieves high validity. Group-wise, GLANCE succeeds more often; T-CREx makes smaller changes but rarely succeeds. The third column adds plausibility: higher log-density means explanations lie in denser regions of the learned distribution. No single method wins across all quality dimensions.

## 1:40-2:00

*Return to the audience and invite a poster conversation.*

CEL makes these trade-offs visible and gives researchers a framework they can extend. At the poster, I can show the full results and how to add a new method. If you work on counterfactual explanations, bring your method to the benchmark.
