import re
import string

def read_wiki(lang, count):
    data = []
    file_path = lang + "\\AA\\wiki_"
    output = open(lang + "_wiki.txt", 'w', encoding='utf-8')
    for i in range(count):
        add = "0" if i < 10 else ""
        for line in open(file_path + add + str(i), encoding='utf-8'):
            # ignore empty lines and html things
            if len(line) == 1 or line[0] == '<':
                continue
        
            sents = line.strip().split('. ')
            for sent in sents:
                if len(sent) != 0:
                    # remove things in the parentheses
                    sent = re.sub(r'\([^()]*\)', '', sent)
                    # cleaning "" and ''
                    sent = re.sub('[“”]', '\"', sent)
                    sent = re.sub('[’]', '\'', sent)
                    # remove non-alphabet / punctuation characters
                    if lang == "af":
                        alphabet = "èêëéîïôöûü"
                    elif lang == "ts":
                        alphabet = ""
                    elif lang == "nso":
                        alphabet = "êôš"
                    allow = string.ascii_letters + string.digits + "\-–,.:\'\"?!% " + alphabet
                    sent = re.sub('[^%s]' % allow, '', sent)
                    print(sent, file = output)
                    data.append(sent)
    output.close()
    return data

if __name__ == '__main__':
    read_wiki("ts", 1)