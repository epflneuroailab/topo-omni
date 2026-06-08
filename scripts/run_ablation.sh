
categories=(faces objects scenes bodies)

# for stimuli in  ${categories[@]}; do
#     echo "Running no ablation with stimuli=${stimuli}"
#     python -m eval.run.run_ablation --stimuli ${stimuli}
# done

# mode=top_p
# for percentage in 30 25 20 15 10 5; do
#     for localizer in ${categories[@]}; do
#         for stimuli in ${categories[@]}; do
#             echo "Running ablation with localizer=${localizer}, stimuli=${stimuli}, percentage=${percentage}, mode=${mode}"
#             python -m eval.run.run_ablation \
#                 --localizer ${localizer} \
#                 --stimuli ${stimuli} \
#                 --mode ${mode} \
#                 --percentage ${percentage} \
#                 --ablate

#             echo "Running stimulation with localizer=${localizer}, stimuli=${stimuli}, percentage=${percentage}, mode=${mode}"
#             python -m eval.run.run_ablation \
#                 --localizer ${localizer} \
#                 --stimuli ${stimuli} \
#                 --mode ${mode} \
#                 --percentage ${percentage} \
#                 --stimulate
#         done
#     done
# done


# mode=top_p
# localizer=faces
# stimuli=faces
# percentage=10
# echo "Running stimulation with localizer=${localizer}, stimuli=${stimuli}, percentage=${percentage}, mode=${mode}"
# python -m eval.run.run_ablation \
#     --localizer ${localizer} \
#     --stimuli ${stimuli} \
#     --mode ${mode} \
#     --percentage ${percentage} \
#     --ablate
# done

# mode=top_p
# localizer=scenes
# stimuli=faces
# percentage=20
# strength=5
# echo "Running stimulation with localizer=${localizer}, stimuli=${stimuli}, percentage=${percentage}, strength=${strength}, mode=${mode}"
# python -m eval.run.run_ablation \
#     --localizer ${localizer} \
#     --stimuli ${stimuli} \
#     --mode ${mode} \
#     --percentage ${percentage} \
#     --stimulate \
#     --stimulation_strength ${strength}
# done