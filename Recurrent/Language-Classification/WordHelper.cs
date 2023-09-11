namespace NeuralNetworks.Recurrent {

    public static class Filter {

        /// <summary>
        /// Static method that filters a file of words with some condition and rewrites it in place.
        /// </summary>
        private static void FilterWords(string readPath, string writePath, Condition condition, int maxWords) {

            string[] words = File.ReadAllLines(readPath);

            // Clear the file if it exists, otherwise create a new one
            if (File.Exists(writePath)) {
                File.WriteAllText(writePath, string.Empty);
            } else {
                var myFile = File.Create(writePath);
                myFile.Close();
            }

            using (StreamWriter writer = new StreamWriter(writePath)) {
                int count = 0;
                for (int i = 0; i < words.Length; i++) {
                    if (condition.Check(words[i])) {
                        writer.WriteLine(ProcessWord(words[i]));
                        count++;
                    }
                    if (count == maxWords) break;
                }
            }
        }

        /// <summary>
        /// Puts the word in lowercase (but more functionality can be added for scalability)
        /// </summary>
        private static string ProcessWord(string word) {
            return word.ToLower();
        }

        public static void FilterFiles() {
            
            string readPathStem  = "Recurrent/Language-Classification/original-words/";
            string writePathStem = "Recurrent/Language-Classification/longer-words/";
            Condition condition = new(5);
            int maxWords = 1000;

            FilterWords(readPathStem + "dutch.txt"  , writePathStem + "dutch.txt"  , condition, maxWords);
            FilterWords(readPathStem + "english.txt", writePathStem + "english.txt", condition, maxWords);
            FilterWords(readPathStem + "french.txt" , writePathStem + "french.txt" , condition, maxWords);
            FilterWords(readPathStem + "german.txt" , writePathStem + "german.txt" , condition, maxWords);
            FilterWords(readPathStem + "spanish.txt", writePathStem + "spanish.txt", condition, maxWords);
            FilterWords(readPathStem + "swedish.txt", writePathStem + "swedish.txt", condition, maxWords);
        }
    }


    // Clearly this class isn't necessary but it makes the program more customisable and scalable
    public class Condition {
        public int minWordLength;
        public int maxWordLength;

        public Condition(int minWordLength = 0, int maxWordLength = Int32.MaxValue) {
            this.minWordLength = minWordLength;
            this.maxWordLength = maxWordLength;
        }

        public bool Check(string word) {

            if (word.Length < minWordLength) return false;
            if (word.Length > maxWordLength) return false;

            return true;
        }
    }
}