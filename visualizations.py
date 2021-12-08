from matplotlib import pyplot as plt


#@param a list_results contains a list of lenght two tuples 
# outer_tuple[0]  is a string describing the parameters used to generate the adversarial text
# outer_tuple[1] is a list of tuples, with inner_tuple[0] as the adv text string, and inner_tuple[1] as the f1 score 
# ex: 
def plot_results(results):
    color_list = ["b","g","r","c","m","k"]
    for i, outer_tuple in enumerate(results):
        description_text_generation_method =outer_tuple[0]
        # inner_tuple = outer_tuple[1]
        # f1_scores = [inner_tuple[i] for i in range(len(inner_tuple))]
        f1_scores = outer_tuple[1]
        length_adv_texts =  list(range(1, len(f1_scores)+1))
        plt.plot(length_adv_texts, f1_scores, 'go-', 
                label = description_text_generation_method,
                color = color_list[i],
                linewidth=2)
    # plt.show()
    # line_1=plt.plot(x1, y1, x2, y2,label="small dataset with evalDataset only why? questions param description 1")
    # plt.show()
    # line1, = plt.plot([1, 2, 3], label="large dataset with ")
    # line2, = plt.plot([3, 2, 1], label="line2")
    leg = plt.legend(loc='upper center')
    plt.title('Adversarial Text Generation with Beam Search')
    plt.xlabel("Universal Trigger Length")
    plt.ylabel("F1 Score")
    plt.show()
# plt.plot(1,0,2,4)
# plt.plot([1, 2, 3], [4, 2, 1], 'go-', label='line 1', linewidth=2)
# plt.plot([1, 2, 3], [1, 4, 9], 'rs', label='line 2')
# #                 length_adv_texts[1],f1_scores[1],
# # plt.set_title('line plot with data points')
# plt.show()
plot_result_param = [("Generated using Small Vocabulary with Why questions Only",  [.8,.7,.6,.65,.68,.5,.2]),
                     ("Generated using large Vocabulary with all questions types",       [.1,.7,.6,.65,.68,.5,.2]),
                     ("Generated using small Vocabulary with all questions types",  [.9,.8,.6,.65,.68,.5,.2])
                    ]

plot_results(plot_result_param)



# # # get lists of ranges to calculate f1_score on to avoid nasty indexing bug
# # ranges_of_eval_dataset = [list(range(50*i,50*i+50)) for i in range(100)]
# # text_with_score = []
# # all_adv_texts = ["Why", "Why Because"]

# # # loop through text
# # for adv_text in all_adv_texts:
# #     scores_without_error = 0
# #     f1_score_sum = 0
# #     # loop through ranges of eval_dataset subsets to calculate f1_score estimate
# #     for range in ranges_of_eval_dataset:
# #         eval_dataset = eval_dataset.select(range)

# #         # calculate f1_score on eval_dataset subset
# #         try:
# #             result = 10 # TODO suket replace with actual evaluate call
# #             f1_score_sum.append(result)
# #             scores_without_error+=1
# #         except Exception as e1:
# #             # if error occurs, ignore the range and continue calculations on next range
# #             continue
# #         # stop loop for adv_text once we have calculated f1_score on 1000 examples
# #         if scores_without_error==20:
# #             avg_f1_score = f1_score_sum/scores_without_error
# #             text_with_score.append((adv_text, avg_f1_score))
# #             break
