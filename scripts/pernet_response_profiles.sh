top_k_pct=${1:-1}
fwhm_mm=${2:-4}
anatomical_constraint=${3:-true}

model_name="qwen2_5_3b_spatial_task_final_7"
# model_name="qwen2_5_3b_task_7"

# for localizer in pernet_fold_A pernet_fold_B; 
# do
#     echo "Evaluating response profiles for ${localizer} localizer"
#     python -m src.eval.analysis.marvi_response_profiles \
#         --model_name ${model_name} \
#         --localizer ${localizer} \
#         --top_k_pct ${top_k_pct} \
#         --fwhm_mm ${fwhm_mm} \
#         --anatomical_constraint ${anatomical_constraint}
# done

python -m src.visualize.plot_pernet_response_profiles \
    --model_name ${model_name} \
    --localizer pernet \
    --top_k_pct ${top_k_pct} \
    --fwhm_mm ${fwhm_mm} \
    --anatomical_constraint ${anatomical_constraint}
