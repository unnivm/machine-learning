
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 
 * This is an implementation of CART, Classification and Regression Trees , is a machine learning algorithm for predictive modeling
 * This is developed in Java and can be used with any other java based application for machine learning purpose. The training and test
 * data are hard coded.
 * 
 * The algorithm builds a decision tree with leaf node as the highest gain achieved. The tree is built recursively.
 * 
 * How it works:
 * ============
 *   
 *   1. First we calculate the Gini impurity of the training data set. You will get more information about Gini impurity from WikiPedia
 *   2. Then calculate the best question and gain based on the data set attributes
 *   3. Based on this best question, will partition the data set and will produce two data sets with true and false conditions
 *   4. Once the decision tree is built, we will predict with test data. 
 * 
 * @author Unni Vemanchery Mana
 *
 */
public class Cart {

	private static List<String> header = new ArrayList<>();

	public static void main(String[] args) {
		new Cart().init();
	}

	/**
	 * 
	 */
	public void init() {
		// adds a header
		setupHeader();

		DataSet dataSet = new DataSet();
		dataSet.prepareTrainingData();

		List<ArrayList<Object>> data = dataSet.getDataSet();
		Node node = buildTree(data, dataSet);

		System.out.println("start printing the tree....");
		printTree(node, " ");

		// now prepare some testing data
		List<ArrayList<Object>> testData = prepareTestData();

		for (ArrayList<Object> row : testData) {
			System.out.println(String.format("Actual: %s, Predicted %s ", row.get(2),
					predict(classify(row, node, dataSet), dataSet)));
		}
	}

	/**
	 * set up column names to be displayed
	 */
	private void setupHeader() {
		header.add("color");
		header.add("diameter");
		header.add("label");
	}

	/**
	 * returns training data
	 * 
	 * @return
	 */
	private List<ArrayList<Object>> prepareTestData() {
		List<ArrayList<Object>> data = new ArrayList<>();

		ArrayList<Object> al = new ArrayList<>();
		al.add("Green");
		al.add(3);
		al.add("Apple");

		data.add(al);

		al = new ArrayList<>();
		al.add("Yellow");
		al.add(3);
		al.add("Apple");

		data.add(al);

		al = new ArrayList<>();
		al.add("Red");
		al.add(1);
		al.add("Grape");

		data.add(al);

		al = new ArrayList<>();
		al.add("Yellow");
		al.add(3);
		al.add("Lemon");

		data.add(al);

		al = new ArrayList<>();
		al.add("Yellow");
		al.add(7);
		al.add("Banana");

		data.add(al);

		al = new ArrayList<>();
		al.add("Green");
		al.add(7);
		al.add("Banana");

		data.add(al);


		al = new ArrayList<>();
		al.add("Orange");
		al.add(2);
		al.add("Orange");

		data.add(al);

		return data;
	}

	/**
	 * Builds a decision tree based on the given training data set.
	 * 
	 * Tree is built recursively by splitting the data set based on
	 * the best question (or criteria). Also, it calculates the current Gini impurity called 'gain'
	 * If the gain is zero , which means that current node is only having the desired item. It does not contain
	 * any other items other than the predicted one. This is the Leaf Node in the decision tree. During the tree building process, will get 
	 * many Leaf nodes. 
	 *
	 * For example, if the data set contains ['Red', 3, 'Apple'] and the gain is zero, then the prediction
	 * is 'Apple' and the Leaf node will be having only this item. This is the best result achieved.
	 * 
	 * Then will partition the data set based on this best question we derived above. It will generate two data sets again, called,
	 * true and false. 
	 * 
	 * With these partitioned data sets (true and false), again will call buildTree function recursively until we get a Leaf node.
	 * This method will return a decision node with best question and the partitioned data sets
	 * 
	 * @param data
	 * @param dataSet
	 * @return Node
	 */
	private Node buildTree(List<ArrayList<Object>> data, DataSet dataSet) {

		List<Object> bestSplit = findBestSplit(data, dataSet);
		
		// gets best gain and question to be asked
		float bestGain = (float) bestSplit.get(0);
		Question bestQuestion = (Question) bestSplit.get(1);
	
		if (bestGain == 0)
			return new LeafNode(data, dataSet);

		//Partition partition = new Partition();
		List<List<ArrayList<Object>>> partitionData = partition(data, bestQuestion);
		List<ArrayList<Object>> trueRows  = partitionData.get(0);
		List<ArrayList<Object>> falseRows = partitionData.get(1);

		// call buildTree with true rows recursively
		Node trueNode = buildTree(trueRows, dataSet);

		// call buildTree with false rows recursively
		Node falseNode = buildTree(falseRows, dataSet);

		return new DecisionNode(bestQuestion, trueNode, falseNode);
	}

	/**
	 * This method returns best question and gain
	 * 
	 * @param data
	 * @param dataSet
	 * @return
	 */
	private List<Object> findBestSplit(List<ArrayList<Object>> data, DataSet dataSet) {
		float impurity = calculateGiniImpurity(data, dataSet);
		int numberOfFeatures = data.get(0).size() - 1; // fetch all column except last one
		float bestGain = 0;
		Question bestQuestion = null;

		for (int col = 0; col < numberOfFeatures; col++) {
			Set<String> unique = dataSet.countUniqueValues(data, col);
			for (String val : unique) {
				Question question = new Question(col, val);

				// partition the data set
				List<List<ArrayList<Object>>> partitionData = partition(data, question);
				if (partitionData.get(0).size() == 0 || partitionData.get(1).size() == 0)
					continue;

				// calculate info gain
				float gain = calculateInfoGain(partitionData.get(0), partitionData.get(1), impurity, dataSet);
				if (gain >= bestGain) {
					bestGain = gain;
					bestQuestion = question;
				}
			}
		}

		// return bestGain and bestQuestion
		List<Object> bestSplit = new ArrayList<>();
		bestSplit.add(bestGain);
		bestSplit.add(bestQuestion);

		return bestSplit;
	}

	/**
	 * Calculates Gini impurity for the given data set
	 * 
	 * @param data
	 * @return
	 */
	private float calculateGiniImpurity(List<ArrayList<Object>> data, DataSet dataSet) {
		Map<String, Integer> counts = dataSet.countAttributes(data);
		float impurity = 1.0f;
		for (Map.Entry<String, Integer> map : counts.entrySet()) {
			String key = map.getKey();
			float probability = counts.get(key) / (float) data.size();
			impurity -= Math.pow(probability, 2);
		}
		return impurity;
	}

	/**
	 * calculates the information gain from the true and false data sets
	 * 
	 * @param left
	 * @param right
	 * @param currentUncertainity
	 * @param dataSet
	 * @return
	 */
	private float calculateInfoGain(List<ArrayList<Object>> left, List<ArrayList<Object>> right,
			float currentUncertainity, DataSet dataSet) {
		float p = left.size() / (right.size() + left.size());
		return currentUncertainity - p * calculateGiniImpurity(left, dataSet)
				- (1 - p) * calculateGiniImpurity(right, dataSet);
	}

	/**
	 * 
	 * @author Unni Mana
	 *
	 */
	private static class DataSet {

		private List<ArrayList<Object>> data = new ArrayList<>();

		/**
		 * 
		 */
		public void prepareTrainingData() {

			ArrayList<Object> al = new ArrayList<>();
			al.add("Green");
			al.add(3);
			al.add("Apple");

			data.add(al);

			al = new ArrayList<>();
			al.add("Yellow");
			al.add(3);
			al.add("Apple");

			data.add(al);

			al = new ArrayList<>();
			al.add("Red");
			al.add(1);
			al.add("Grape");

			data.add(al);

			al = new ArrayList<>();
			al.add("Yellow");
			al.add(3);
			al.add("Lemon");

			data.add(al);

			al = new ArrayList<>();
			al.add("Yellow");
			al.add(7);
			al.add("Banana");

			data.add(al);

			al = new ArrayList<>();
			al.add("Green");
			al.add(7);
			al.add("Banana");
			data.add(al);

			al = new ArrayList<>();
			al.add("Orange");
			al.add(3);
			al.add("Orange");
			data.add(al);

			System.out.println(data);
			countAttributes(data);
		}

		/**
		 * counts the frequency of unique attributes in the data set
		 * For example data set is ['green', 3, 'Apple'], ['yellow', 3, 'Orange']. Then this method will create a map with the frequency
		 * of attributes
		 * 
		 * @param data
		 */
		public Map<String, Integer> countAttributes(List<ArrayList<Object>> data) {
			Map<String, Integer> counts = new HashMap<>();
			String label;
			
			for (ArrayList<Object> al : data) {
				label   = (String) al.get(al.size() - 1); // gets the last column value (apple or orange etc)
				int cnt = counts.get(label) == null ? 0 : counts.get(label);
				cnt++;
				counts.put(label, cnt);
			}
			
			return counts;
		}

		/**
		 * 
		 * @param data
		 * @param col
		 * @return
		 */
		public Set<String> countUniqueValues(List<ArrayList<Object>> data, int col) {
			Set<String> set = new HashSet<>();
			for (ArrayList<Object> al : data) {
				Object colObj = al.get(col);
				set.add(colObj.toString());
			}

			return set;
		}

		public List<ArrayList<Object>> getDataSet() {
			return data;
		}

	}

	/**
	 * 
	 * @author Unni Vemanchery Mana
	 *
	 * Decision node with question and true and false branch
	 */
	private static class DecisionNode implements Node {

		public Question question;
		private Node trueBranch;
		private Node falseBranch;

		public DecisionNode(Question question, Node trueBranch, Node falseBranch) {

			this.question    = question;
			this.trueBranch  = trueBranch;
			this.falseBranch = falseBranch;
		}
	}

	/**
	 * 
	 * @author Unni Vemanchery Mana
	 *
	 * Holds the data set attribute  and its position
	 */
	private static class Question {

		private int col;
		private String val;

		public Question(int col, String val) {
			this.col = col;
			this.val = val;
		}

		/**
		 * match the data set attributes based on the value and column number
		 *  
		 * @param example
		 * @return
		 */
		public boolean match(ArrayList<Object> example) {
			if (example.get(col) instanceof String) {
				return this.val == example.get(col);
			}
			int columnValue = (Integer) example.get(col);
			return Integer.parseInt(this.val) >= columnValue;
		}

		@Override
		public String toString() {
			String condition = "==";
			if (col == 1)
				condition = ">=";
			String format = String.format("Is %s %s %s ", header.get(col), condition, val);
			return format;
		}
	}

	
		/**
		 * Partition data set based on the question that matched
		 * 
		 * @param rows
		 * @param question
		 * @return
		 */
		public List<List<ArrayList<Object>>> partition(List<ArrayList<Object>> rows, Question question) {
			List<ArrayList<Object>> trueRows = new ArrayList<>();
			List<ArrayList<Object>> falseRows = new ArrayList<>();
			List<List<ArrayList<Object>>> parition = new ArrayList<>();
			for (ArrayList<Object> row : rows) {
				if (question.match(row)) {
					trueRows.add(row);
				} else
					falseRows.add(row);
			}
			parition.add(trueRows);
			parition.add(falseRows);
			
			return parition;
		}
	

	/**
	 * 
	 * @author Unni Vemanchery Mana
	 *
	 * Leaf Node of the decision tree. This class is primarily responsible for holding the predictions
	 * 
	 */
	private static class LeafNode implements Node {

		private Map<String, Integer> predictions = new HashMap<>();

		public LeafNode(List<ArrayList<Object>> rows, DataSet dataSet) {	
			predictions = dataSet.countAttributes(rows);
		}
	}

	/**
	 * This is a marker interface to identify a type
	 * 
	 * @author Unni Mana
	 *
	 */
	public interface Node {
	}

	/**
	 * calls this method recursively to print the decision tree
	 * 
	 * @param node
	 * @param space
	 */
	private void printTree(Node node, String space) {

		if (node instanceof LeafNode) {
			LeafNode ln = (LeafNode) node;
			System.out.println(space + " Predict " + ln.predictions);
			return;
		}

		DecisionNode dNode = (DecisionNode) node;

		System.out.println(space + dNode.question);
		System.out.println(space + "---> True ");

		printTree(dNode.trueBranch, space + " ");

		System.out.println(space + "---> False ");
		printTree(dNode.falseBranch, space + " ");
	}

	/**
	 * method will get the predictions
	 * 
	 * @param counts
	 * @param dataSet
	 * @return
	 */
	private Map<String, String> predict(Map<String, Integer> counts, DataSet dataSet) {
		Map<String, String> probs = predict(counts);
		return probs;
	}

	/**
	 * This method will calculate the prediction in percentage of each unique item in the test data with
	 * confidence level 1.
	 * 
	 * @param counts
	 * @return
	 */
	private Map<String, String> predict(Map<String, Integer> counts) {
		float total = 0;
		int sum = 0;
		Map<String, String> probs = new HashMap<>();
		for (Map.Entry<String, Integer> map : counts.entrySet()) {
			sum += map.getValue();
		}
		total = sum * 1.0f;
		
		for (Map.Entry<String, Integer> map : counts.entrySet()) {
			String key = map.getKey();
			probs.put(key, ((int) (counts.get(key) / total * 100) + "%"));
		}
		
		return probs;
	}

	/**
	 * This method will classify the data based on the trained data set and the test data.
	 * The output of this method will return the predictions.
	 * 
	 * @param row
	 * @param node
	 * @param dataSet
	 * @return
	 */
	private Map<String, Integer> classify(ArrayList<Object> row, Node node, DataSet dataSet) {
		if (node instanceof LeafNode) {
			LeafNode ln = (LeafNode) node;
			return ln.predictions;
		}
		DecisionNode dNode = (DecisionNode) node;
		if (dNode.question.match(row))
			return classify(row, dNode.trueBranch, dataSet);

		return classify(row, dNode.falseBranch, dataSet);
	}

}
