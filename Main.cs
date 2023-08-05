using System;
using System.Collections.Generic;
using System.Linq;

namespace RecurrentNeuralnetwork {

    public class Language {

        enum Languages { Dutch, English, French, German, Italian, Spanish };
        
        static Random rng = new Random();

        public static void Main() {
            TestLSTM();
        }
    
        public static void TestUtility() {
            // everything seems to work properly
            double[][] matrix0 = new[] {
                new double[] {1, 2, 3},
                new double[] {4, 5, 6},
                new double[] {7, 8, 9},
                new double[] {9, 10, 11}
            };
            double[,] matrix1 = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {9, 10, 11}
            };
            double[,] matrix2 = {
                {10, 11, 12},
                {13, 14, 15},
                {16, 17, 17},
                {18, 19, 20}
            };
            // Matrix.Display(matrix0[^1]);
            // Matrix.Display(Matrix.Add(matrix1, matrix2));
            
            double[] vector1 = {1, 4, 7};
            double[] vector2 = {0.5, 1, 1.5};
            double[] vector3 = {0.9, 1, 1.1};
            // Matrix.Display(Matrix.Add(vector1, vector2));
            // Matrix.Display(Matrix.Add(vector1, vector2, vector3));
            // Matrix.Display(Matrix.Add(matrix1, 5));
            // Matrix.Display(Matrix.Add(vector1, 1.2));
            // Matrix.Display(Matrix.ScalarMultiply(matrix1, 2));
            // Matrix.Display(Matrix.ScalarMultiply(vector1, 1.5));

            double[,] matrix3 = {
                {10, 11, 12, 20, 21},
                {13, 14, 15, 4, 1},
                {16, 17, 17, 9, 2},
            };

            // Matrix.Display(Matrix.MatrixMultiply(matrix1, matrix3));
            // Matrix.Display(Matrix.MatrixMultiply(matrix1, vector2));
            double[] vector4 = {-2, 1, 6, -1.5};

            // Matrix.Display(Matrix.MatrixMultiply(vector1, vector4));
            // Matrix.Display(Matrix.Transpose(matrix3));

            // Matrix.Display(Activation.Sigmoid(vector4));
            // Matrix.Display(Derivative.Sigmoid(vector4));

            Matrix.Display(Matrix.Concatenate(vector1, vector4));
        }

        public static void TestNetwork() {
            RNN rnn = new RNN("networks/LanguageClassification.txt");

            string word = "0";
            while (word != "q") {
                Console.Write("Enter a word: ");
                word = Console.ReadLine().ToLower();
                var dictionary = GetVocabDictionary();
                int[] input = ConvertWordToInput(word, dictionary);
                Console.WriteLine((Languages)rnn.Predict(input));
            }

        }

        public static void TrainLanguages(int epochs) {
            Dictionary<char, int> vocabDictionary = GetVocabDictionary();
            List<(string , int)> data = GetLanguageData();

            RNN rnn = new RNN(vocabDictionary.Count, 12, 6);
            // RNN rnn = new RNN("networks/LanguageClassification.txt");

            for (int i = 0; i < epochs; i++) {
                Shuffle(data);

                int[] correct = new int[6];
                int[] total = new int[6];


                foreach ((string word, int label) pair in data) {
                    int[] input = ConvertWordToInput(pair.word, vocabDictionary);
                    if (rnn.Train(input, pair.label)) {
                        correct[pair.label]++;
                    }
                    total[pair.label]++;
                }

                double[] successRate = new double[6];
                for (int j = 0; j < 6; j++) {
                    successRate[j] = Math.Round(correct[j] * 100.0 / total[j], 2);
                }

                Console.WriteLine($"Epoch {i+1}:");
                Console.WriteLine($"Dutch: {successRate[0]}%");
                Console.WriteLine($"English: {successRate[1]}%");
                Console.WriteLine($"French: {successRate[2]}%");
                Console.WriteLine($"German: {successRate[3]}%");
                Console.WriteLine($"Italian: {successRate[4]}%");
                Console.WriteLine($"Spanish: {successRate[5]}%");
                Console.WriteLine($"Overall: {Math.Round(correct.Sum() * 100.0 / total.Sum(), 2)}%");
                Console.WriteLine();

            }

            // Save the network
            rnn.SaveNetwork("networks/LanguageClassification.txt");

        }

        public static List<(string, int)> GetLanguageData() {
            List<(string, int)> data = new();

            string[] lines;

            lines = File.ReadAllLines("words/dutch.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.Dutch));
            }

            lines = File.ReadAllLines("words/english.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.English));
            }

            lines = File.ReadAllLines("words/french.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.French));
            }

            lines = File.ReadAllLines("words/german.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.German));
            }

            lines = File.ReadAllLines("words/italian.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.Italian));
            }

            lines = File.ReadAllLines("words/spanish.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.Spanish));
            }


            return data;
        }

        public static HashSet<char> GetVocabulary(List<(string, int)> data) {
            HashSet<char> chars = new();
            foreach ((string, int) pair in data) {
                foreach (char c in pair.Item1) {
                    chars.Add(Char.ToLower(c));
                }
            }

            return chars;
        }

        public static Dictionary<char, int> GetVocabDictionary() {
            Dictionary<char, int> dictionary = new();
            int value = 0;
            string[] lines = File.ReadAllLines("words/characters.txt");
            foreach (string str in lines) {
                dictionary.Add(char.Parse(str), value);
                value++;
            }
            return dictionary;

        }

        public static int[] ConvertWordToInput(string word, Dictionary<char, int> dictionary) {
            int[] res = new int[word.Length];
            for (int i = 0; i < word.Length; i++) {
                res[i] = dictionary[char.ToLower(word[i])];
            }
            return res;
        }


        public static void TestLSTM() {
            // LSTM lstm = new LSTM(26, 10, 6);
            LSTM lstm = new LSTM("networks/lstm.txt");
            int[] inputs = {4, 6, 1, 9, 10};
            lstm.Train(inputs, 4);
            lstm.Train(inputs, 4);
            lstm.SaveNetwork("networks/lstm.txt");

        }



        // Fisher-Yates shuffle
        public static void Shuffle<T>(IList<T> list)  
        {  
            int n = list.Count;  
            while (n > 1) {  
                n--;
                int k = rng.Next(n + 1);
                (list[n], list[k]) = (list[k], list[n]);
            }
        }

    }
}