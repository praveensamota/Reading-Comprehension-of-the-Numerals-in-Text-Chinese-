Reading-Comprehension-of-the-Numerals-in-Text-
Reading Comprehension of Numerals in Text (Chinese) Introduction Reading comprehension of numerals in text is a vital aspect of natural language processing (NLP), with applications across various domains such as medicine, engineering, finance, and more. This project focuses on developing a model capable of accurately interpreting numerical information embedded within Chinese textual data. Understanding and extracting these numerals is essential for tasks like financial analysis and medical diagnosis, where precise numerical comprehension can significantly impact the quality of outputs.

Motivation The main motivation behind this project is to enable machines to understand and interpret numerical data within natural language. Accurate prediction and representation of numerical information are crucial in NLP tasks like text summarization and headline generation. For instance, in news summarization, the correct inclusion of statistics or percentages can enhance the informativeness and relevance of the generated summaries. This project aims to improve the machine's ability to comprehend and handle numerical data, thereby advancing NLP applications.

Methodology Data Preparation Template Creation: The project begins with setting random seed values for reproducibility. A Template class instance is created to manage input and label templates. The templates include placeholders for dynamic content, such as numerical values and answer options, which are stored in a dictionary for easy access during data preprocessing. Dataset Loading and Preprocessing: The dataset, containing over 70,000 questions from the NQuAD (Numerical Question Answering Dataset), is loaded from a JSON file. The data is cleaned, and relevant information is extracted. Tokenization is then applied to convert the textual data into a numerical format suitable for machine learning models. Model Training The train_model function fine-tunes a pre-trained T5-small model using the Hugging Face Transformers library. The model is trained using the Seq2SeqTrainer, which simplifies the training process by handling data collation, setting training arguments, and iterating over batches. The choice of the T5-small model is due to its effectiveness in handling various NLP tasks, including those requiring input-output mappings like text summarization and question answering.

Evaluation The testing phase involves evaluating the model on a separate dataset, calculating metrics like micro and macro F1 scores, option accuracy, and numerical accuracy. The evaluation also includes visualizations such as confusion matrices and precision-recall curves to gain insights into the model's performance. The results are stored in a JSON file for further analysis.

Use of CUDA CUDA is utilized in this project to accelerate the training process by leveraging GPU capabilities for parallel computation. This is especially beneficial given the model's need to process large amounts of data and perform complex calculations. The project uses PyCUDA for managing CUDA-related tasks and optimizing model training speed and efficiency. This setup allows for faster iterations during training, enabling the model to learn more effectively from the data.

Dataset The NQuAD dataset, designed for fine-grained numeracy testing, includes questions that require selecting the correct numeral based on given text. This dataset tests machine comprehension models' ability to handle numerals, a critical aspect of tasks like headline generation. NQuAD provides a challenging test environment with over 70,000 questions and closely related numerical options, pushing the boundaries of current NLP models.

Discussion The model showed improvement over the training epochs, with decreasing training and validation losses and increasing numerical and option accuracy. While these results are promising, there is still room for improvement. Future work could explore using more advanced models like BERT or Google-mT5, or techniques like ensembling and transfer learning to achieve better results.

Conclusion This project demonstrated a reasonable level of performance in understanding numerals in Chinese text using a T5-small model. However, there is potential for further improvements, particularly in refining numerical accuracy calculations and exploring other model architectures. The project serves as a foundation for future work in numeracy comprehension in NLP, with the potential to significantly enhance various applications across multiple domains.

Installation and Usage To replicate this project, follow these steps:

Install Required Libraries:

Python 3.8 or higher PyTorch Hugging Face Transformers PyCUDA (for CUDA support)

Data Preparation:

Download the NQuAD dataset. Run the preprocessing script to prepare the data. Model Training:

Fine-tune the T5-small model using the provided training script. Evaluation:

Evaluate the model on the test dataset and analyze the results.

About
