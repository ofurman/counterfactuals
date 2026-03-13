# uv run scripts/generate_latex_tables_single_dataset.py --output small-tables/results_wine_local.tex --include-dataset wine --exclude-methods TCREx,GlobalGLANCE,GroupGLANCE,AReS,GLOBE_CE --drop-empty-rows

# uv run scripts/generate_latex_tables_single_dataset.py --output small-tables/results_law_local.tex --include-dataset law --exclude-methods TCREx,GlobalGLANCE,GroupGLANCE,AReS,GLOBE_CE  --drop-empty-rows

# uv run scripts/generate_latex_tables_single_dataset.py --output small-tables/results_adult_census_group.tex --include-dataset adult_census --include-methods TCREx,GroupGLANCE --drop-empty-rows --discriminative-model MLP

# uv run scripts/generate_latex_tables_single_dataset.py --output small-tables/results_gmsc_global.tex --include-dataset give_me_some_credit --include-methods AReS,GLOBE_CE,GlobalGLANCE --drop-empty-rows --discriminative-model MLP
