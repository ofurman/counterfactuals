
# uv run scripts/generate_latex_tables.py --output tables/results_categorical.tex --exclude-methods TCREx,GlobalGLANCE,GroupGLANCE,AReS,GLOBE_CE --exclude-datasets blobs,digits,moons,wine,audit,heloc --drop-empty-rows

uv run scripts/generate_latex_tables.py --output tables/results_numerical.tex --exclude-methods TCREx,GlobalGLANCE,GroupGLANCE,AReS,GLOBE_CE --include-datasets blobs,digits,moons,wine,audit,heloc

# uv run scripts/generate_latex_tables.py --output tables/results_group.tex --include-methods TCREx,GroupGLANCE
#
# uv run scripts/generate_latex_tables.py --output tables/results_global.tex --include-methods GlobalGLANCE,AReS,GLOBE_CE
