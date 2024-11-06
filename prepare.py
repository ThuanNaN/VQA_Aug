import argparse
import pandas as pd
from tqdm import tqdm
from utils import seed_everything, DataPath
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paraphraser_tokenizer = MT5Tokenizer.from_pretrained(
    "chieunq/vietnamese-sentence-paraphase",
    legacy=False)
paraphraser_model = MT5ForConditionalGeneration.from_pretrained(
    "chieunq/vietnamese-sentence-paraphase").to(device)


def get_paraphrase(text, num_return_sequences, max_length=64):
    inputs = paraphraser_tokenizer(text,
                                   padding='longest',
                                   max_length=64,
                                   return_tensors='pt',
                                   return_token_type_ids=False).to(device)

    outputs = paraphraser_model.generate(inputs.input_ids,
                                        attention_mask=inputs.attention_mask,
                                        max_length=max_length,
                                        num_beams=num_return_sequences,
                                        early_stopping=True,
                                        no_repeat_ngram_size=1,
                                        num_return_sequences=num_return_sequences)

    paraphrase_lst = []
    for beam_output in outputs:
        paraphrase_lst.append(
            paraphraser_tokenizer.decode(beam_output,
                                         skip_special_tokens=True)
        )

    return paraphrase_lst


def get_sorted_paraphrase(paraphrase_with_scores: list, from_index: int, topK: int):
    paraphrase_with_scores_sorted = sorted(paraphrase_with_scores, key=lambda x: x[1], reverse=True)
    paraphrase_with_scores_sorted = [para for para, score in paraphrase_with_scores_sorted]
    filtered_para_questions = paraphrase_with_scores_sorted[from_index:(from_index + topK)]
    return filtered_para_questions


def knn_filter(origin, para_lst, from_index, topK):
    collection = [origin] + para_lst
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(collection)
    scores = cosine_similarity(tfidf_matrix[:1], tfidf_matrix[1:]).flatten()
    paraphrase_with_scores = list(zip(para_lst, scores))
    return get_sorted_paraphrase(paraphrase_with_scores, from_index, topK)


def sbert_filter(origin, para_lst, from_index, topK):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    paraphrase_with_scores = []

    for para in para_lst:
        score = util.cos_sim(model.encode(origin), model.encode(para)).item()
        paraphrase_with_scores.append((para, score))
    return get_sorted_paraphrase(paraphrase_with_scores, from_index, topK)


def generate_paraphrases(data_filepath, num_paraphrase, filter_method, from_index, topk):
    df = pd.read_csv(data_filepath)
    filtered_question_paraphrases = []
    for idx, row in tqdm(df.iterrows(), desc='Generating paraphrases', total=len(df)):
        question_paraphrases = get_paraphrase(row['question'], num_paraphrase)
        if filter_method == 'knn':
            filtered_question_paraphrases.append(knn_filter(row['question'], question_paraphrases, from_index, topk))
        elif filter_method == 'sbert':
            filtered_question_paraphrases.append(sbert_filter(row['question'], question_paraphrases, from_index, topk))
        else:
            filtered_question_paraphrases.append(question_paraphrases)

    df['question_paraphrase'] = filtered_question_paraphrases
    return df


def generate_answer_space():
    train_csv_path = DataPath.ViVQA_PATH / 'train.csv'
    test_csv_path = DataPath.ViVQA_PATH / 'test.csv'

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_answers = train_df['answer'].unique().tolist()
    test_answers = test_df['answer'].unique().tolist()
    
    answer_space = set(list(train_answers + test_answers))

    save_path =  DataPath.ViVQA_PATH / 'answer_space.txt'

    with open(save_path, 'w+') as f:
        f.write('\n'.join(answer_space))

def main():
    parser = argparse.ArgumentParser(description="Generate a new dataset")
    parser.add_argument("--random_seed", type=int,
                        default=59, help="Random seed")
    parser.add_argument("--train_filepath", type=str, required=True,
                        help="Path to the training dataset (train.csv)")
    parser.add_argument("--save_path", type=str,
                        required=True, help="Path to save the dataset")
    parser.add_argument("--num_paraphrase", type=int,
                        default=20, help="Number of num_paraphrase")
    parser.add_argument("--filter_method", type=str, default='knn',
                        help="Select a filtering method: 'knn', 'sbert', 'no' (default: 'knn')")
    parser.add_argument("--from_index", type=int, default=0,
                        help="Select the start index to get paraphrases from after sorted in descending cosine similarity order (default: 0)")
    parser.add_argument("--topk", type=int, default=10,
                        help="Select the top-k results (default: 10)")
    args = parser.parse_args()

    seed_everything(args.random_seed)

    paraphrase_df = generate_paraphrases(data_filepath=args.train_filepath,
                                         num_paraphrase=args.num_paraphrase,
                                         filter_method=args.filter_method,
                                         from_index=args.from_index,
                                         topk=args.topk)

    paraphrase_df.to_csv(args.save_path, index=False)
    print(f'Paraphrase results saved to {args.save_path}')

    generate_answer_space()
    print('Answer space generated')

if __name__ == "__main__":
    main()
