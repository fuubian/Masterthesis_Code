from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import os
import sys

# Variables and directories
workspace_dir = "/pfs/work7/workspace/scratch/ma_frajwa-dataset/"
qa_output_dir = workspace_dir + "qa_output/"

table_dir = workspace_dir + "extracted_tables/"
table_image_dir = table_dir + "table_images/"
table_code_dir = table_dir + "table_code/"
table_metadata = table_dir + "tables.csv"

figure_dir = workspace_dir + "extracted_figures/"
figure_metadata = figure_dir + "figures.csv"

os.makedirs(qa_output_dir, exist_ok=True)

# Define QA-Pair-Generation Prompts
table_prompt = """
Table: {table_code} 
Caption: {caption} 
Text mentions: {text_mentions} 
Question: [Your generated question here] 
Answer: [Your generated answer here]"""

figure_prompt = """
Caption: {caption} 
Text mentions: {text_mentions} 
Question: [Your generated question here] 
Answer: [Your generated answer here]"""

few_shot_learning_table = """
Generate an open-ended question and its corresponding answer based on a scientific table. Use its caption and text mentions
from the scientific paper to create a question that tests the understanding of this specific table and answer it afterwards. 
The answer should be unambigious and consist of as few words as possible.

Caption: Experimental results on the ISIDCM-20 dataset. ``Mean" denotes the average CT value in the selected ROIs. ``AR" denotes the absolute error between the average CT value of each method and that of referenced NDCT in the selected ROIs. The best and second-best performance in each column is colorized by the \textcolor{red}{red} and the  \textcolor{blue}{blue}.
Text mentions: \subsubsection{Performance comparison on clinical dataset} We follow \cite{10081080} to choose the two most representative ROIs for performance comparison on clinical datasets, as shown in Figures \ref{real1} and \ref{real2}. We calculate the average CT Hounsfield units value and corresponding absolute error with reference images (unpaired NDCT images) as used in \cite{10081080}.  As illustrated in Table \ref{isicdm-20}, quantitative results demonstrate that our proposed method has a closer CT value with the reference image. For example, in ROI-2, the average CT value of our proposed method is -25.15, instead, the second-best result achieved by CCDnet is only -13.91. By observing visualized results in Figure \ref{real1}, it seems that most baseline methods (especially for Noiser2noise and CycleGAN) achieve an unfavorable noise suppression for this very challenging LDCT image with complex and serious noise.
Question: What is the second-best average CT value in ROI-2?\
Answer: -13.91.

Caption: \label{tab:pkg-times} Comparison of different \proglang{Python} packages for Bayesian optimization. The elapsed time per iteration averaged across the ten runs is given for each package.
Text mentions: However, the time \pkg{NUBO} requires to complete one iteration with a maximum of 2.20s for D) is, on average, higher than for the other packages (Table~\ref{tab:pkg-times}). While this might be important for some areas of optimization, it is negligible when optimizing expensive black-box functions, as these functions are much more resource-intensive to evaluate. Thus, the small number of additional seconds that \pkg{NUBO} requires per iteration is insignificant compared to the resources required to conduct an experiment or a simulation.
Question: Which package has the highest elapsed time per iteration in scenario D, and what is its maximum recorded time?
Answer: NUBO; 2.20s.

Caption: Change area estimates for Tigray region and each zone within Tigray (units are kha)
Text Mentions: Table \ref{tab:change-area} summarizes the area estimates for the four transition classes. We found that cropland gain accounted for $70-160$ kha while cropland loss accounted for $19-79$ kha in Tigray. The stable transition classes accounted for the majority of the area in Tigray: $1,033-1,293$ kha of stable cropland and $3,798-4,054$ kha of stable non-cropland.
Question: What is the estimated range of area for stable cropland in the Tigray region?
Answer: 1,033-1,293 kha.

Caption: \label{tab:heterogeneity}Heterogeneity Analysis 
Text Mentions: Table \ref{tab:heterogeneity} presents the results of a heterogeneity analysis using the sample of users with an organisational affiliation. The results reveal that the adverse effects of the ban on productivity are mainly driven by users who created their profile prior to 2016 and users with 15 followers or less. While the latter could be an indicator for less skilled users, the former results suggests that older users are more impacted by the ban.
Question: Which two user groups are primarily driving the adverse effects of the ban on productivity?
Answer: Users who created their profile prior to 2016 and users with 15 followers or less.

Caption: Experiment 2 Results: Summary of the median, mean, and standard deviation for the epochs required to reach the threshold value across 1000 simulation runs, and a counter of faster convergence runs between models.
Text Mentions: The principal findings of the second experiment are presented in Table~\ref{tab:Iterations}, providing a comprehensive comparison of the epochs required to achieve the predefined loss value of $0.05$, as outlined in Section~\ref{setup2}, between the SCQRNN model and the CQRNN baseline model.
Question: Which predefined loss value is used as the threshold for comparing the epochs required for convergence between the SCQRNN and CQRNN models?
Answer: 0.05.

"""

few_shot_learning_figure = """
Generate an open-ended question and its corresponding answer based on a scientific figure. Use its caption and text mentions
from the scientific paper to create a question that tests the understanding of this specific figure and answer it afterwards. 
The answer should be unambigious and consist of as few words as possible.

2209.01769_FIG_4
Caption: \textcolor{black}{Rate-distortion comparison of GOP sizes 4, 8, 16 on UVG dataset under intra-period 32.} 
Text mentions: We evaluate the effect of GOP size on the performance of B-CANF. A number of GOP sizes, including 4, 8, 16, are tested with intra-period 32. The BD-rates are summarized in Table~\ref{tab:abl_gop} (see the results w/o a separate P-frame codec). The corresponding rate-distortion curves on UVG dataset are presented in Fig.~\ref{fig:abl_gop_size}. From Fig.~\ref{fig:abl_gop_size}, the rate-distortion performance of B-CANF is seen to improve with the increased GOP size. The improvement is most obvious at low rates. Like P-frames, our B*-frames suffer more from temporal error propagation with smaller GOP sizes (in which cases, B*-frames are sent more frequently), especially at low rates where poor reconstruction and motion quality is expected. Increasing GOP size decreases the frequency of B*-frames, thereby reducing temporal error propagation.
Question: How does increasing the GOP size affect the rate-distortion performance of B-CANF on the UVG dataset under intra-period 32?
Answer: Increasing GOP size improves rate-distortion performance.

2203.08550_FIG_7
Caption: The finger contribution comparison of the bending angle for human and robot hands. (a) Single-direction grasp, (b) Bidirectional grasp.
Text Mentions: Based on the collected data of the fingers of the soft robot and human hands, the proportion of the bending angle were calculated to analyze the contribution of each fingers. As shown in Figure \ref{IROSFigBiomimeticRatioComparsion}, for the single-direction grasp, the thumb, index and middle fingers act as the main roles for the grasp pose. Beside, the relative high weight of the human ring finger is caused by the [missing part]
Question: Which fingers contribute the most to the bending angle in a single-direction grasp for both human and robot hands?
Answer: Thumb, index, and middle fingers.

2206.07171_FIG_2
Caption: Categorization of the 38 papers reviewed in this survey. The papers are first categorized on the learning paradigm (fully vs. semi/un/self-supervised) and on the segmentation type (semantic vs. instance). Each quadrant shows the distributions of applications (2D vs. 3D) and DL backbones (U-Net vs. FCN vs. Other) of the papers that use the corresponding learning and segmentation approaches. 
Text mentions: Fig.~\ref{fig:SearchResultSummary} summarizes this collection of 38 papers in terms of learning technique (fully supervised or not), segmentation type (semantic or instance), application (2D or 3D) and the underlying modeling backbone. Before reviewing these papers, we discuss the key EM datasets and describe the evolution of DL architectures, which are two crucial components that have been permitting the progress of EM segmentation analysis.
Question: How are the 38 papers reviewed in the survey categorized?
Answer: The papers are categorized by learning paradigm (fully vs. semi/un/self-supervised) and segmentation type (semantic vs. instance).

2201.06313_FIG_4
Caption: The general architecture of the proposed method 
Text mentions: In designing the model for the proposed method, the SoftMax function must be used in the output layer of the model because each category has three different classes of positive, negative, and neutral. Since our number of categories is 9, 9 SoftMax functions with three neurons were used. Fig. \ref{fig:4} shows the general architecture of the proposed method based on hard parameter sharing to solve the two sub-tasks of aspect category detection and aspect category polarity for joint learning.
Question: How is the output layer of the general architecture of the proposed method designed?
Answer: The output layer uses 9 SoftMax functions with three neurons each for the categories: positive, negative, and neutral.

2205.10981_FIG_5
Caption: GPT-3 Classification Endpoint mean performance with standard errors on data science question topic classification. Training data is augmented by adding different quantities of new examples generated with GPT-3 Davinci Completion. 
Text mentions: While, for the validation set, accuracy was positively related to the number of questions generated, the same was not true for the test set. Figure \ref{res:fig-class} plots the relationship between accuracy and number of example questions added across both validation and test sets, with the shaded regions representing the standard error (68 percent confidence interval). Note that the x-axis is a log scale. On the test set, accuracy scarcely increased at all until the number of questions added reached about 1,000, at which point it increased to 76 percent. This represented peak accuracy; augmented training sets with 10,000 new questions averaged only 73 percent accuracy, a slight drop.
Question: How does augmenting training data with GPT-3-generated examples affect test set accuracy in data science question topic classification?
Answer: Test set accuracy increases slightly after adding about 1,000 examples, peaking at 76%, but decreases slightly to 73% with 10,000 examples."""

# Generate a qa_pair, either for a figure or a table
def generate_qa_pair(object_id, image_file, caption, text_mentions, table_code, model, tokenizer):
    # Modifying the prompt
    task_input_prompt = None
    instruction_prompt = None
    if table_code:
        instruction_prompt = few_shot_learning_table
        task_input_prompt = table_prompt.replace("{caption}", caption).replace("{text_mentions}", text_mentions).replace("{table_code}", table_code)
    else:
        instruction_prompt = few_shot_learning_figure
        task_input_prompt = figure_prompt.replace("{caption}", caption).replace("{text_mentions}", text_mentions)
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that can understand and generate question-answer pairs from scientific data."},
        {"role": "user", "content": f"{instruction_prompt}\n{task_input_prompt}"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Receiving the results and store them in file
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response
    
# Read either figure or table data from csv file
def get_object_data(meta_file, start_index, end_index, table=False):
    object_data = {}
    with open(meta_file, "r", newline='', encoding='utf-8') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        index = 0
        for row in spamreader:
            if index >= start_index:
                object_id = row[0]
                caption = row[3]
                text_mentions = row[4]

                if table: # For tables
                    try:
                        table_code = get_table_code(table_code_dir + object_id + ".tex")
                        object_data[object_id] = (caption, text_mentions, table_code)
                    except Exception as e:
                        print(f"Error occurred for index {index}: {e}")
                else: # For figures
                    object_data[object_id] = (caption, text_mentions)
            index += 1
            if index > end_index:
                break

    return object_data

# Return table code
def get_table_code(code_file):
    table_code = None
    if os.path.isfile(code_file):
        with open(code_file, "r", encoding='utf-8') as code_file:
            table_code = code_file.read()
            splitted_code = table_code.split("\pagenumbering{gobble}")
            if len(splitted_code) != 2:
                raise ValueError(f"Unexpected occurrence of pagenumbering. Please check manually {code_file}")
            table_code = splitted_code[-1]
            table_code = table_code.replace("\end{document}", "")
    else:
        raise FileNotFoundError(f"{code_file} was not found.")
    return table_code

# Execute whole QA generation for either figures or tables, following a range of indexes in the metadata file
def execute_generation(start_index, end_index, table=False):
    # Loading model
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Data extraction from csv file started.")
    object_data = None
    try:
        if table:
            object_data = get_object_data(table_metadata, start_index, end_index, True)
        else:
            object_data = get_object_data(figure_metadata, start_index, end_index)
    except Exception as e:
        print("Error during data extraction:", e)
        return None

    print("QA-pair generation started.")
    counter = 0
    for obj in object_data:
        image_file = None
        table_code = None
        caption = object_data[obj][0]
        text_mentions = object_data[obj][1]
        if table:
            image_file = table_image_dir + obj + ".png"
            table_code = object_data[obj][2]
        else:
            image_file = figure_image_dir + obj + ".png"
            
        # Generation
        try:    
            response = generate_qa_pair(obj, image_file, caption, text_mentions, table_code, model, tokenizer)
        except Exception as e:
            print(e)
        else:
            output_file = qa_output_dir + obj + ".txt"
            with open(output_file, "w", encoding='utf-8') as output:
                output.write(response)

        counter += 1
        if counter % int(len(object_data)/10) == 0:
            print(f"{counter} objects have been processed.")

    print("Process complete.")

# Main function
def main(s_index, e_index, is_table):
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    execute_generation(s_index, e_index, is_table)
   
# Running main function   
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3:
        print(f"Unexpeceted number of arguments received. Expected: 3; Received: {len(args)}")
    else:
        try:
            s_index = int(args[0])
            e_index = int(args[1])
            is_table = args[2].lower() == "true"
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            execute_generation(s_index, e_index, is_table)