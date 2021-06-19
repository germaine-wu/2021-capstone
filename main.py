
from config import parseArgs
from prediction import prediction
from validation import evaluate


def main():
    args = parseArgs()
    function = args.function
    input_file = args.input
    output_file = args.output
    true_file = args.trueout
    result_file = args.result
    method = args.method

    if function == ("Pred" or "Predict"):
        if input_file == "":
            print("Error - Read inputs")
        elif output_file == "":
            print("Error - Write output")
        else:
            prediction(input_file, output_file, method)
    elif function == ("Eval" or "Evaluate"):
        if input_file == "" or true_file == "":
            print("Error - Read inputs")
        elif output_file == "":
            print("Error - Write output")
        else:
            evaluate(input_file, output_file, true_file, result_file)
    else:
        print("Please choose 'Pred' function or 'Eval' function")


if __name__ == '__main__':
    main()
