top_k_pct=${1:-10}
fwhm_mm=${2:-8.0}

localizers=(language_text theory_of_mind_text multiple_demand_text)

for localizer in "${localizers[@]}";
do
    # for odd_even in odd even; 
    # do
    #     echo "Evaluating response profiles for ${localizer} localizer, odd_even=${odd_even}..."
    #     python -m src.eval.analysis.cognitive_response_profiles --localizer ${localizer} --odd_or_even ${odd_even} --top_k_pct ${top_k_pct} --fwhm_mm ${fwhm_mm}
    # done
    python -m src.visualize.plot_cog_response_profiles --localizer ${localizer} --top_k_pct ${top_k_pct}
done

# odd_even="odd"
# localizer="language_text"
# python -m eval.a`nalysis.cognitive_response_profiles --localizer ${localizer} --odd_or_even ${odd_even} --top_k_pct ${top_k_pct}
