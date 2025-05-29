To re-run this experimental setup, all the necessary files are present in the setup folder. Tested setup folder is 
test run I did locally and contains all generated prompts, responses, classifier scores, manual annotations, visualizations produced.

For the scripts to run, following Python libraries must be installed:
os, csv, pandas, tqdm, matplotlib, seaborn, transformer, torch, sklearn.metrics,jinja2



prompt_generator cript an be adjusted to use any other model of choice from hugging-face- LLaMa, Mistral. Here Phi-2 was used.
When running the script using terminal, it automatically downloads and caches the model first time provided valid token is
already provided in the terminal.

Seed folder contains txt files. These prompts were found to be effective at getting LMs to output harmful responses. Adjust them
to user's liking.

Catalytic prompts are just advanced version of these prompts, can be used experimentally if vanilla prompts aren't effective enough.
These are basically jailbreaks and confuse LMs safety measures.

Run the classifier script which adds a toxicity score after each response in a csv file in a separate folder.

Do manual annotation to csv generated responses by adding either 0 or 1 after the toxicity core assigned  the classifier.

Then run the analyzer script. This asks for a threshold score for a response to be classified as toxic. My setup used a toxicity score of 0.08 which was quite low s classifier itself wasn't effective in detecting subtle nuances and PHI-2 itself Is academically oriented
and doesn't stray much from it.

All the confusion matrix, scatter plots will be then saved automatically in a separate directory.

For easy of use, a separate html generator script is  provided which generates an html file for easy viewing of all plots, graphs, F1 scores, Precision, Recall and Accuracy.

Thank You, for any questions regarding the setup, contact at :prakamyawasthi_ec24a18_007@dtu.ac.in
