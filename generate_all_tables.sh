uv run generate_latex_tables.py --model MultinomialLogisticRegression
uv run generate_latex_tables.py --model MultilayerPerceptron 
uv run generate_latex_tables.py --model NODE

[ -f tables_combined.tex ] && rm tables_combined.tex

cat main_metrics_mnlogisticregression.tex >> tables_combined.tex
echo "" >> tables_combined.tex

cat distances_mnlogisticregression.tex >> tables_combined.tex
echo "" >> tables_combined.tex

cat main_metrics_mlperceptron.tex >> tables_combined.tex
echo "" >> tables_combined.tex

cat distances_mlperceptron.tex >> tables_combined.tex
echo "" >> tables_combined.tex

cat main_metrics_node.tex >> tables_combined.tex
echo "" >> tables_combined.tex

cat distances_node.tex >> tables_combined.tex
echo "" >> tables_combined.tex

echo "combined table generated"
