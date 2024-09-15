using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Recurrent {

    public static class Classify {

        enum Languages { Dutch, English, French, German, Spanish, Swedish };

        static string path = "Recurrent/Language-Classification/networks/longer-words-network.txt";
        static Dictionary<char, int> dictionary;

        static Classify() {
            dictionary = GetVocabDictionary();
        }
        
        public static void TestNetwork() {
            Vanilla rnn = new Vanilla(path);

            string word = "?";
            while (word != "") {
                Console.Write("Enter a word: ");
                word = Console.ReadLine().ToLower();
                int[] input = ConvertWordToInput(word, dictionary);
                Console.WriteLine((Languages)rnn.Predict(input));
            }
        }

        public static void TrainLanguages(int epochs = 10, bool saveNetwork = false) {
            Dictionary<char, int> vocabDictionary = GetVocabDictionary();
            List<(string , int)> data = GetLanguageData();

            // Split into training data and test data
            var splitData = Data.TrainTestSplit(data, 0.2);
            var trainData = splitData[0];
            var testData  = splitData[1];

            Vanilla rnn = new Vanilla(vocabDictionary.Count, 12, 6);
            // Vanilla rnn = new Vanilla(path);

            // LSTM lstm = new LSTM(vocabDictionary.Count, 12, 6);

            int[] correct;
            int[] total;
            double[] successRate;

            // Train on the training data
            for (int i = 0; i < epochs; i++) {
                Utility.Shuffle(trainData);

                correct = new int[6];
                total = new int[6];

                foreach ((string word, int label) pair in trainData) {
                    int[] input = ConvertWordToInput(pair.word, vocabDictionary);
                    if (rnn.Train(input, pair.label)) {
                        correct[pair.label]++;
                    }
                    total[pair.label]++;
                }

                successRate = new double[6];
                for (int j = 0; j < 6; j++) {
                    successRate[j] = Math.Round(correct[j] * 100.0 / total[j], 2);
                }

                Console.WriteLine($"Epoch {i+1}:");
                Console.WriteLine($"Dutch: {successRate[0]}%");
                Console.WriteLine($"English: {successRate[1]}%");
                Console.WriteLine($"French: {successRate[2]}%");
                Console.WriteLine($"German: {successRate[3]}%");
                Console.WriteLine($"Spanish: {successRate[4]}%");
                Console.WriteLine($"Swedish: {successRate[5]}%");
                Console.WriteLine($"Overall: {Math.Round(correct.Sum() * 100.0 / total.Sum(), 2)}%");
                Console.WriteLine();
            }

            // Test on the test data
            correct = new int[6];
            total = new int[6];
            foreach ((string word, int label) pair in testData) {
                int[] input = ConvertWordToInput(pair.word, vocabDictionary);
                if (rnn.PredictAndCheck(input, pair.label)) {
                    correct[pair.label]++;
                }
                total[pair.label]++;
            }

            successRate = new double[6];
            for (int j = 0; j < 6; j++) {
                successRate[j] = Math.Round(correct[j] * 100.0 / total[j], 2);
            }

            Console.WriteLine();
            Console.WriteLine("--------------- Test data ---------------");
            Console.WriteLine($"Dutch: {successRate[0]}%");
            Console.WriteLine($"English: {successRate[1]}%");
            Console.WriteLine($"French: {successRate[2]}%");
            Console.WriteLine($"German: {successRate[3]}%");
            Console.WriteLine($"Spanish: {successRate[4]}%");
            Console.WriteLine($"Swedish: {successRate[5]}%");
            Console.WriteLine($"Overall: {Math.Round(correct.Sum() * 100.0 / total.Sum(), 2)}%");
            Console.WriteLine();

            // Save the network
            if (saveNetwork) {
                rnn.SaveNetwork(path);
            }
        }

        public static List<(string, int)> GetLanguageData() {
            List<(string, int)> data = new();

            string pathStem = "Recurrent/Language-Classification/longer-words/";

            string[] lines;

            lines = File.ReadAllLines(pathStem + "dutch.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.Dutch));
            }

            lines = File.ReadAllLines(pathStem + "english.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.English));
            }

            lines = File.ReadAllLines(pathStem + "french.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.French));
            }

            lines = File.ReadAllLines(pathStem + "german.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.German));
            }

            lines = File.ReadAllLines(pathStem + "spanish.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.Spanish));
            }

            lines = File.ReadAllLines(pathStem + "swedish.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.Swedish));
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

        public static void WriteVocabDictionaryToFile(string path) {

            HashSet<char> chars = GetVocabulary(GetLanguageData());

            // Clear the file if it exists, otherwise create a new one
            if (File.Exists(path)) {
                File.WriteAllText(path, string.Empty);
            } else {
                var myFile = File.Create(path);
                myFile.Close();
            }

            using (StreamWriter writer = new StreamWriter(path)) {
                foreach (char c in chars) {
                    writer.WriteLine(c);
                }
            }
        }

        public static Dictionary<char, int> GetVocabDictionary() {
            Dictionary<char, int> dictionary = new();
            int value = 0;
            string[] lines = File.ReadAllLines("Recurrent/Language-Classification/longer-words/characters.txt");
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
    
        public static void PrepareData() {
            string dictionaryPath = "Recurrent/Language-Classification/longer-words/characters.txt";
            Filter.FilterFiles();
            WriteVocabDictionaryToFile(dictionaryPath);
        }
    }
}