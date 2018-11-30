import java.io.*;
import java.util.ArrayList;

/**
 * An InstanceSet is a set of instances to be used in a machine learning
 * problem, generally as either a training set or a test set.
 * 
 * @author jmac
 * 
 */
public class InstanceSet {
	// All instances in the InstanceSet share the same set of
	// attributes, stored here as an AttributeSet
	private AttributeSet attributeSet = new AttributeSet();

	// An ArrayList containing all instances in the InstanceSet
	private ArrayList<Instance> instances = new ArrayList<Instance>();

	/**
	 * The character used to start comments in .arff files.
	 */
	public static final String commentStart = "%";

	/**
	 * Construct an InstanceSet by reading a .arff file with the given filename
	 * 
	 * @param inputFilename
	 *            name of the file to read
	 * @throws IOException
	 * @throws DecisionTreeException
	 * @throws IOException 
	 * @throws Exception
	 */
	public InstanceSet(String inputFilename) throws DecisionTreeException, IOException {
		parseInputFile(inputFilename);
	}

	/**
	 * Construct an InstanceSet from a list of instances
	 * 
	 * @param attributeSet
	 *            the set of attributes for this set of instances
	 * @param instances
	 *            a list of the instances to be stored in the instance set
	 */
	public InstanceSet(AttributeSet attributeSet, ArrayList<Instance> instances) {
		super();
		this.attributeSet = attributeSet;
		this.instances = instances;
	}

	private void parseInputFile(String inputFilename)
			throws DecisionTreeException, IOException {
		BufferedReader reader = new BufferedReader(
				new FileReader(inputFilename));

		// Input file comprises a "preamble" followed by "data".
		// Data segment is demarcated by a line consisting of the string
		// "@data".

		// We make everything case insensitive by transforming to
		// lowercase before doing anything else.

		// We parse the preamble first, constructing attributes and
		// the like.
		String line = reader.readLine().toLowerCase();
		line = line.toLowerCase();
		while (!parsePreambleLine(line)) {
			line = reader.readLine();
			line = line.toLowerCase();
		}
		attributeSet.setDefaultClassAttribute();

		System.out.println("\nFinished processing preamble, attribute set is:");
		this.attributeSet.print();

		// Now parse the data
		line = reader.readLine().toLowerCase();
		while (line != null) {
			line = line.toLowerCase();
			parseDataLine(line);
			line = reader.readLine();
		}
		reader.close();

		System.out.println("\nFinished processing data, " + instances.size()
				+ " instances found:");
		for (Instance instance : instances)
			instance.print();
		System.out.println();
	}

	private boolean shouldIgnoreLine(String line) {
		if (line.equals(""))
			return true;
		else if (line.startsWith(commentStart))
			return true;
		else if (line.startsWith("@relation"))
			return true;
		else
			return false;
	}

	// Return true if this line is the start of the data segment,
	// otherwise return false
	private boolean parsePreambleLine(String line) throws DecisionTreeException {
		if (shouldIgnoreLine(line))
			return false;
		else if (line.startsWith("@attribute")) {
			parseAttribute(line);
			return false;
		} else if (line.startsWith("@data"))
			return true;
		else
			throw new DecisionTreeException("unexpected line: " + line);
	}

	private void parseAttribute(String line) {
		// split on whitespace to extract attribute name
		String[] split_line = line.split("\\s+");
		String attribute_name = split_line[1];

		// Extract the part in braces, which is the list of possible
		// attribute values.
		// To keep things simple, we will split on all brace
		// characters, thus assuming that the only place either
		// brace character appears is at the start and end of the
		// attribute value list
		split_line = line.split("[{}]");
		String value_list = split_line[1];

		// To get the actual values out of value_list, split on commas
		// and whitespace
		String[] values = value_list.split("[,\\s]+");

		// Construct the attribute and add it to the attribute set
		Attribute attribute = new Attribute(attribute_name, values);
		this.attributeSet.addAttribute(attribute);
	}

	private void parseDataLine(String line) {
		if (shouldIgnoreLine(line))
			return;

		// Line of data values should be comma-separated, so split on
		// commas and whitespace
		String[] values = line.split("[,\\s]+");
		Instance instance = new Instance(values);
		this.instances.add(instance);
	}

	/**
	 * Get the set of attributes used by all the instances in this instance set.
	 * @return the attributeSet
	 */
	public AttributeSet getAttributeSet() {
		return attributeSet;
	}

	/**
	 * Get a list of all instances in this instance set.
	 * @return the instances
	 */
	public ArrayList<Instance> getInstances() {
		return instances;
	}

	/**
	 * Get the number of instances in this instance set.
	 * @return the number of instances in this set of instances
	 */
	public int getNumInstances() {
		return instances.size();
	}
}