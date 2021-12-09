from matplotlib import pyplot as plt


#@param a list_results contains a list of lenght two tuples 
# outer_tuple[0]  is a string describing the parameters used to generate the adversarial text
# outer_tuple[1] is a list of tuples, with inner_tuple[0] as the adv text string, and inner_tuple[1] as the f1 score 
# ex: 
def plot_results(results, why_questions):
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
    # leg = plt.legend(loc='right')
    plt.legend(loc=(.4, .55))
    plt.title('Adversarial Text Generation with Beam Search')
    plt.xlabel("Universal Trigger Length")
    if why_questions:
        plt.ylabel("F1 Score Calculated on \'Why\' Questions")
    else:
        plt.ylabel("F1 Score Calculated on \n['Why', 'What', 'When', 'Where', 'How'] Questions")
    plt.show()

# plot_result_param = [("Generated using Small Vocabulary with Why questions Only",  [.8,.7,.6,.65,.68,.5,.2]),
#                      ("Generated using large Vocabulary with all questions types",       [.1,.7,.6,.65,.68,.5,.2]),
#                      ("Generated using small Vocabulary with all questions types",  [.9,.8,.6,.65,.68,.5,.2])
#                     ]


# evaluated_multiple_question_types = [   ("Generated using Small Vocabulary with Why questions Only",    [82.43, 80.06, 78.30, 77.56, 77.96, 77.99, 77.10, 77.33, 76.81, 77.02]),
#         ("Generated using Large Vocabulary with All questions types",   [82.43, 84.02, 82.97, 80.12, 78.46]                         ),
#         ("Generated using Small Vocabulary with All questions types",   [82.43, 81.77, 83.92, 83.64, 83.75,83.8, 83.68]             ),
#         ("Randomly Generated Words",                                    [82.43, 83.60, 83.92, 83.73, 83.96, 83.81]                  ),
#         ("Wallace Suggested Universal Trigger",                        [82.43, 82.51, 82.19, 81.75]                                 )
#     ]

# evaluated_why_questions_only = [   ("Generated using Small Vocabulary with Why questions Only",    [66.11, 54.98, 48.48, 43.45, 46.14, 46.50, 43.81, 42.68, 43.11, 41.59]),
#         ("Generated using Large Vocabulary with All questions types",   [66.11, 72.88, 67.68, 55.00, 48.93]                         ),
#         ("Generated using Small Vocabulary with All questions types",   [66.11, 69.74, 68.52, 58.64, 53.75, 53.81, 53.65]           ),
#         ("Randomly Generated Words",                                    [66.11, 71.68, 72.23, 72.21, 72.18, 72.18]                  ),
#         ("Wallace Suggested Universal Trigger",                        [66.11, 65.01, 64.42, 62.07]                                 )
#     ]
evaluated_multiple_question_types = [   ("Generated using Small Vocabulary with Why questions Only",    [82.43, 80.06, 78.30, 77.56, 77.96, 77.99]),
        ("Generated using Large Vocabulary with All questions types",   [82.43, 84.02, 82.97, 80.12, 78.46]                         ),
        ("Generated using Small Vocabulary with All questions types",   [82.43, 81.77, 83.92, 83.64, 83.75,83.8]             ),
        ("Randomly Generated Words",                                    [82.43, 83.60, 83.92, 83.73, 83.96, 83.81]                  ),
        ("Wallace Suggested Universal Trigger",                        [82.43, 82.51, 82.19, 81.75]                                 )
    ]

evaluated_why_questions_only = [   ("Generated using Small Vocabulary with Why questions Only",    [66.11, 54.98, 48.48, 43.45, 46.14, 46.50]),
        ("Generated using Large Vocabulary with All questions types",   [66.11, 72.88, 67.68, 55.00, 48.93]                         ),
        ("Generated using Small Vocabulary with All questions types",   [66.11, 69.74, 68.52, 58.64, 53.75, 53.81]           ),
        ("Randomly Generated Words",                                    [66.11, 71.68, 72.23, 72.21, 72.18, 72.18]                  ),
        ("Wallace Suggested Universal Trigger",                        [66.11, 65.01, 64.42, 62.07]                                 )
    ]

plot_results(evaluated_why_questions_only, why_questions = True)
plot_results(evaluated_multiple_question_types, why_questions = False)