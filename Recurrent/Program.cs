using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Recurrent {

    public static class Program {

        enum Languages { Dutch, English, French, German, Italian, Spanish };
        
        public static void TestNetwork() {
            Vanilla rnn = new Vanilla("Recurrent/networks/LanguageClassification.txt");

            string word = "0";
            while (word != "q") {
                Console.Write("Enter a word: ");
                word = Console.ReadLine().ToLower();
                var dictionary = GetVocabDictionary();
                int[] input = ConvertWordToInput(word, dictionary);
                Console.WriteLine((Languages)rnn.Predict(input));
            }

        }

        public static void TrainLanguages(int epochs = 10) {
            Dictionary<char, int> vocabDictionary = GetVocabDictionary();
            List<(string , int)> data = GetLanguageData();

            Vanilla rnn = new Vanilla(vocabDictionary.Count, 12, 6);
            // RNN rnn = new RNN("networks/LanguageClassification.txt");

            LSTM lstm = new LSTM(vocabDictionary.Count, 12, 6);
            // LSTM lstm = new LSTM("networks/lstm.txt");

            for (int i = 0; i < epochs; i++) {
                Utility.Shuffle(data);

                int[] correct = new int[6];
                int[] total = new int[6];

                // for (int j = 0; j < 10; j++) {
                //     (string word, int label) pair = data[j];
                //     int[] input = ConvertWordToInput(pair.word, vocabDictionary);
                //     lstm.Train(input, pair.label);
                // }

                foreach ((string word, int label) pair in data) {
                    int[] input = ConvertWordToInput(pair.word, vocabDictionary);
                    if (lstm.Train(input, pair.label)) {
                    // if (rnn.Train(input, pair.label)) {
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
            // rnn.SaveNetwork("networks/LanguageClassification.txt");
            lstm.SaveNetwork("Recurrent/networks/lstm.txt");

        }

        public static List<(string, int)> GetLanguageData() {
            List<(string, int)> data = new();

            string[] lines;

            lines = File.ReadAllLines("Recurrent/words/dutch.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.Dutch));
            }

            lines = File.ReadAllLines("Recurrent/words/english.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.English));
            }

            lines = File.ReadAllLines("Recurrent/words/french.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.French));
            }

            lines = File.ReadAllLines("Recurrent/words/german.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.German));
            }

            lines = File.ReadAllLines("Recurrent/words/italian.txt");
            foreach (string line in lines) {
                data.Add((line, (int)Languages.Italian));
            }

            lines = File.ReadAllLines("Recurrent/words/spanish.txt");
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
            string[] lines = File.ReadAllLines("Recurrent/words/characters.txt");
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
    }
}