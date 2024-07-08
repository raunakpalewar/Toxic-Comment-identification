
import ai21



def paraphrase(comment):

    ai21.api_key = ('x62I8ZV78hONc5QWuOsRUw3wOJCDsIzm')

    response = ai21.Paraphrase.execute(text=comment, max_returned_sequences=1)
    result = ''
    for suggestion in response['suggestions']:
        # suggestion = response['suggestions'][1]['text']
        result = result + suggestion['text']+'\n'

    file0 = open("flask_php\\static\\para_result.txt", 'r+')
    file0.truncate(0)
    file0.close()

    file1 = open("flask_php\\static\\para_result.txt",'a')
    file1.write(result)
    print(result)
    # Closing file
    file1.close()

if __name__ == "__main__":
    p = paraphrase('i AM not Well.')

