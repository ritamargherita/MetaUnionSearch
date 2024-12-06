import random
import sys

def select_random_topics(file_path, random_seed):
    random.seed(random_seed)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header = lines[0]
    topics = lines[1:]
    num_to_select = int(len(topics) * 0.4)
    topics_test_set = random.sample(topics, num_to_select)
    topics_eval_set = [topic for topic in topics if topic not in topics_test_set]
    return header, topics_test_set, topics_eval_set

def save_topics(header, topics, output_path):
    with open(output_path, 'w') as file:
        file.write(header)
        file.writelines(topics)
    return

def main(input_file, random_seed, test_output_file, eval_output_file):
    """
    """
    header, topics_test_set, topics_eval_set = select_random_topics(input_file, random_seed)
    save_topics(header, topics_test_set, test_output_file)
    save_topics(header, topics_eval_set, eval_output_file)
    return


if __name__ == "__main__":

    input_file = "../data/topics_50.txt"
    test_output_file = "../data/test/topics_test_set.txt"
    eval_output_file = "../data/evaltopics_eval_set.txt"
    random_seed = 42
    main(input_file, random_seed, test_output_file, eval_output_file)