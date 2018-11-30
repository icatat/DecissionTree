import java.lang.reflect.Array;
import java.util.ArrayList;

/**
 * Represents a leaf in a decision tree.
 * 
 * @author jmac
 * 
 */
public class DecisionTreeLeaf extends DecisionTree {

	// the decision that this leaf node always makes
	String decision;

	/**
	 * This protected constructor cannot be called by external code; decision
	 * trees should be constructed using the constructDecisionTree factory
	 * method in the DecisionTree class.
	 * 
	 * @param examples
	 *            The examples from which this tree should be learned.
	 * @param label
	 *            The label on the edge leading to this DecisionTree node, or
	 *            <code>DecisionTree.ROOT_LABEL</code> for the root.
	 * @param depth
	 *            The depth of this node in the full decision tree.
	 */
	protected DecisionTreeLeaf(InstanceSet examples, String label, int depth) {
		super(label, depth);
		decision = computeDecision(examples);
	}

	/**
	 * @param examples
	 *            The set of examples from which to compute the decision
	 * @return The decision that this leaf node will make on the given set of
	 *         examples.
	 */
	private String computeDecision(InstanceSet examples) {

		AttributeSet attributes = examples.getAttributeSet();
		Attribute classAttr = attributes.getClassAttribute();
		int classAttrIndex = attributes.getClassAttributeIndex();
		Distribution distr = new Distribution(classAttr);

		ArrayList<Instance>instances = examples.getInstances();
		for (int i = 0; i < instances.size(); i++) {
			distr.incrementFrequency(instances.get(i).getValues()[classAttrIndex]);
		}
		return distr.getNameOfMaxFrequency();

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see DecisionTree#decide(AttributeSet, Instance)
	 */
	@Override
	public String decide(AttributeSet attributes, Instance instance) {
		return decision;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see DecisionTree#print()
	 */
	@Override
	public void print() {
		super.print();
		System.out.println("[decision " + decision + "]");
	}
}
