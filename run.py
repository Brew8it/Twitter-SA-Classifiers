from TSA.Classifiers.NB import NaiveBayes
from TSA.Classifiers.SVM import SupportVectorMachine
from TSA.Classifiers.CNN import ConvolutionalNeuralNetwork


def main_menu():
    print("Welcome to Brew8it and Selberget TSA program")
    print("Please choose the menu you want to start:")
    print("1 test Naive Bayes")
    print("2 test SVM")
    print("3 test CNN")
    print("0 Exit")
    choise = input()
    exec_menu(choise)


def exec_menu(choise):
    if choise == "1":
        model = NaiveBayes()
        print(model.predict("I never asked Cornet for Personal Loyalty. I hardly even knew this guy. Just another of his many lies. His \"memos\" are self serving and FAKE!"))

    if choise == "2":
        model = SupportVectorMachine()
        print(model.predict("I never asked Cornet for Personal Loyalty. I hardly even knew this guy. Just another of his many lies. His \"memos\" are self serving and FAKE!"))
    if choise == "3":
        model = ConvolutionalNeuralNetwork()
        print(model.predict("I never asked Cornet for Personal Loyalty. I hardly even knew this guy. Just another of his many lies. His \"memos\" are self serving and FAKE!"))



if __name__ == '__main__':
    main_menu()