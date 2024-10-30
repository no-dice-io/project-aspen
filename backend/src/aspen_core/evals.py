""" evals.py
All evaluation functions/classes are located here, and will be used in adjacent scripts for evaluation purposes within DSPy. 
"""
import dspy

class CorrectnessJudge(dspy.Signature):
    """Judge if the answer is factually correct based on the context and question."""
    
    context = dspy.InputField(desc="Original context/query that was asked")
    question = dspy.InputField(desc="Question that was asked to the LLM")
    answer = dspy.InputField(desc="Answer provided by the LLM")
    factually_correct = dspy.OutputField(
        desc="Return 1 if the answer is factually correct, 0 if not",
        type=int
    )

class RelevanceJudge(dspy.Signature):
    """Judge if the answer is relevant to the original question."""
    
    question = dspy.InputField(desc="Original question asked to the LLM")
    answer = dspy.InputField(desc="Answer provided by the LLM")
    relevance_score = dspy.OutputField(
        desc="Return an integer from 1 to 5 indicating relevance",
        type=int
    )

class JudgingSystem: # revisit this later
    def __init__(self):
        self.correctness_judge = dspy.ProgramOfThought(CorrectnessJudge)
        self.relevance_judge = dspy.ProgramOfThought(RelevanceJudge)
    
    def evaluate_response(self, context, question, answer):
        # Evaluate correctness
        correctness = self.correctness_judge(
            context=context,
            question=question,
            answer=answer
        )
        
        # Evaluate relevance
        relevance = self.relevance_judge(
            question=question,
            answer=answer
        )
        
        return {
            'is_correct': correctness.factually_correct,
            'relevance_score': relevance.relevance_score
        }

    def correctness_metric(self, example, pred):
        """Metric function for correctness optimization"""
        factual = self.correctness_judge(
            context=example.context,
            question=example.question,
            answer=pred.answer
        )
        return factual.factually_correct  # Already an integer

    def relevance_metric(self, example, pred):
        """Metric function for relevance optimization"""
        relevance = self.relevance_judge(
            question=example.question,
            answer=pred.answer
        )
        return relevance.relevance_score / 5.0  # Normalize to 0-1

'''
lm = dspy.LM('openai/gpt-4o-2024-08-06')
dspy.configure(lm=lm)


# Initialize the judging system
judges = JudgingSystem()

# Example usage
context = "The sky appears blue due to Rayleigh scattering."
question = "Why is the sky blue?"
answer = "The sky appears blue because of a phenomenon called Rayleigh scattering."

# Evaluate a response
evaluation = judges.evaluate_response(context, question, answer)

print(f"Correctness: {evaluation['is_correct']}")
print(f"Relevance Score: {evaluation['relevance_score']}/5")

'''