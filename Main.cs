using System;
using System.Collections.Generic;
using System.Linq;

namespace RecurrentNeuralnetwork {

    public class Language {

        enum Languages { Dutch, English, French, German, Italian, Spanish };
        
        static Random rng = new Random();

        public static void Main() {
            // TrainLanguages(1);
            TestNetwork();
        }
    
        public static void Test() {
            double[,] matrix1 = {
                {0, 4},
                {1, 5},
                {2, 6}
            };
            double[,] matrix2 = {
                {1, 2, 3, 4},
                {5, 6, 7, 8}
            };

            double[,] matrix3 = {
                {0, 1},
                {2, 3},
                {4, 5}
            };

            // Matrix.Display(Matrix.MatrixMultiply(matrix1, matrix2));

            // Matrix.Display(Matrix.Subtract(matrix1, matrix3));
            // Console.WriteLine();
            // Matrix.Display(matrix1);


            double[] vector1 = {3, 5, 6};
            double[] vector2 = {1, 6.5, 2};
            double[] vector = {2, 3};

            // Matrix.Display(Matrix.MatrixMultiply(matrix3, vector));

            // Matrix.Display(Matrix.Transpose(matrix1));

            double[] soft = Activation.Softmax(vector1);
            double[] tanh = Activation.Tanh(vector1);

            // foreach (double s in tanh) {
            //     Console.WriteLine(s);
            // }


            double[] vector3 = {0.15, 0.23, 0.62};
            double[] vector4 = {0.9, 0.1, 0};
            // Console.WriteLine(Loss.CrossEntropy(vector3, vector4));

            int[] vector5 = {0, 4, 6, 2, 1};
            double[] vector6 = {0, 0, 0, 0, 0, 0};

            Matrix.Display(vector1);
            Matrix.Display(vector2);

            vector1 = Matrix.Add(vector1, vector2);
            Matrix.Display(vector1);
            Matrix.Display(vector2);
        }
    

        public static void TestNetwork() {
            RNN rnn = new RNN("networks/LanguageClassification.txt");
        }

        public static void TrainLanguages(int epochs) {
            Dictionary<char, int> vocabDictionary = GetVocabDictionary();
            List<(string , int)> data = GetLanguageData();

            RNN rnn = new RNN(vocabDictionary.Count, 20, 10);

            for (int i = 0; i < epochs; i++) {
                Shuffle(data);

                foreach ((string word, int label) pair in data) {
                    int[] input = ConvertWordToInput(pair.word, vocabDictionary);
                    rnn.Train(input, pair.label);
                }
            }

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