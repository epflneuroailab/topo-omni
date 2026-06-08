top_k_pct=${1:-1}
fwhm_mm=${2:-4}
anatomical_constraint=${3:-true}

model_name="qwen2_5_3b_spatial_task_final_7"
# model_name="qwen2_5_3b_task_7"

localizers=(faces scenes objects vwfa bodies speech)

for localizer in "${localizers[@]}"; 
do
    # for odd_even in odd even; 
    # do
    #     echo "Evaluating response profiles for ${localizer} localizer, odd_even=${odd_even}..."
    #     python -m src.eval.analysis.marvi_response_profiles \
    #         --model_name ${model_name} \
    #         --localizer ${localizer} \
    #         --odd_or_even ${odd_even} \
    #         --top_k_pct ${top_k_pct} \
    #         --fwhm_mm ${fwhm_mm} \
    #         --anatomical_constraint ${anatomical_constraint}
    # done

    python -m src.visualize.plot_response_profiles \
        --model_name ${model_name} \
        --localizer ${localizer} \
        --top_k_pct ${top_k_pct} \
        --fwhm_mm ${fwhm_mm} \
        --anatomical_constraint ${anatomical_constraint}
done