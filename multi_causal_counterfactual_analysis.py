import torch
import numpy as np
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import seaborn as sns

@dataclass
class GroundTruthMetrics:
    reliability_score: float  # How reliable the prediction is (0-1)
    uncertainty: float       # Model's uncertainty measure
    pn: float               # Necessity probability
    ps: float               # Sufficiency probability
    pns: float              # Combined causal effect
    confidence_region: str   # High/Medium/Low confidence classification

@dataclass
class SequenceAnalysis:
    text: str
    token_metrics: List[GroundTruthMetrics]
    overall_reliability: float
    critical_points: List[int]  # Points where intervention might be needed
    high_confidence_regions: List[Tuple[int, int]]  # Spans of reliable predictions

class GroundTruthAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.confidence_threshold = 0.8
        self.uncertainty_threshold = 0.3
        
    def calculate_reliability_score(
        self,
        probs: torch.Tensor,
        entropy_val: float,
        top_k_consistency: float
    ) -> float:
        """Calculate a reliability score based on multiple factors."""
        max_prob = torch.max(probs).item()
        prob_spread = (torch.sort(probs, descending=True)[0][:5] - 
                      torch.sort(probs, descending=True)[0][5:10]).mean().item()
        
        # Combine factors with learned weights
        reliability = (
            0.4 * max_prob +  # High confidence in top prediction
            0.3 * (1 - entropy_val/np.log(len(probs))) +  # Low uncertainty
            0.3 * top_k_consistency  # Consistent top predictions
        )
        return max(0.0, min(1.0, reliability))

    def calculate_causal_metrics(
        self,
        base_probs: torch.Tensor,
        intervention_probs: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Calculate necessity and sufficiency probabilities."""
        # Probability that intervention was necessary
        pn = 1 - torch.cosine_similarity(base_probs, intervention_probs, dim=0).item()
        
        # Probability that intervention was sufficient
        ps = torch.max(intervention_probs).item()
        
        # Combined probability
        pns = pn * ps
        
        return pn, ps, pns

    def classify_confidence_region(
        self,
        reliability: float,
        uncertainty: float
    ) -> str:
        """Classify the confidence region based on metrics."""
        if reliability > self.confidence_threshold and uncertainty < self.uncertainty_threshold:
            return "High"
        elif reliability < 0.3 or uncertainty > 0.7:
            return "Low"
        else:
            return "Medium"

    def analyze_sequence(
        self,
        input_text: str,
        max_steps: int = 100
    ) -> SequenceAnalysis:
        """Perform detailed ground truth analysis of generated sequence."""
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        sequence = input_ids.clone()
        
        token_metrics = []
        critical_points = []
        high_confidence_spans = []
        span_start = None
        
        for step in range(max_steps):
            with torch.no_grad():
                outputs = self.model(input_ids=sequence)
                probs = torch.nn.functional.softmax(outputs.logits[:, -1, :], dim=-1)[0]
            
            # Calculate basic metrics
            entropy_val = entropy(probs.detach().numpy())
            top_k_probs = torch.topk(probs, k=10)[0]
            top_k_consistency = (top_k_probs[0] - top_k_probs[-1]).item()
            
            # Calculate reliability score
            reliability = self.calculate_reliability_score(
                probs, entropy_val, top_k_consistency
            )
            
            # Calculate uncertainty
            uncertainty = entropy_val / np.log(len(probs))
            
            # Calculate causal metrics with hypothetical intervention
            intervention_probs = self.simulate_intervention(probs)
            pn, ps, pns = self.calculate_causal_metrics(probs, intervention_probs)
            
            confidence_region = self.classify_confidence_region(reliability, uncertainty)
            
            metrics = GroundTruthMetrics(
                reliability_score=reliability,
                uncertainty=uncertainty,
                pn=pn,
                ps=ps,
                pns=pns,
                confidence_region=confidence_region
            )
            token_metrics.append(metrics)
            
            if reliability < self.confidence_threshold:
                critical_points.append(step)
            
            if confidence_region == "High":
                if span_start is None:
                    span_start = step
            elif span_start is not None:
                high_confidence_spans.append((span_start, step-1))
                span_start = None
            
            next_token = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
            sequence = torch.cat([sequence, next_token], dim=1)
        
        overall_reliability = np.mean([m.reliability_score for m in token_metrics])
        
        return SequenceAnalysis(
            text=self.tokenizer.decode(sequence[0], skip_special_tokens=True),
            token_metrics=token_metrics,
            overall_reliability=overall_reliability,
            critical_points=critical_points,
            high_confidence_regions=high_confidence_spans
        )

    def simulate_intervention(self, probs: torch.Tensor) -> torch.Tensor:
        """Simulate an intervention by modifying probabilities."""
        intervention_probs = probs.clone()
        top_k = torch.topk(intervention_probs, k=5)
        intervention_probs[top_k.indices] *= 0.8 
        intervention_probs = torch.nn.functional.softmax(intervention_probs, dim=0)
        return intervention_probs

    def visualize_analysis(
        self,
        analysis: SequenceAnalysis,
        save_path: Optional[str] = None
    ):
        """Create visualization of ground truth analysis."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        steps = range(len(analysis.token_metrics))
        
        reliability_scores = [m.reliability_score for m in analysis.token_metrics]
        uncertainty_scores = [m.uncertainty for m in analysis.token_metrics]
        
        ax1.plot(steps, reliability_scores, 'b-', label='Reliability')
        ax1.plot(steps, uncertainty_scores, 'r-', label='Uncertainty')
        ax1.set_title('Model Reliability and Uncertainty')
        ax1.set_ylabel('Score')
        ax1.legend()
        
        pn_scores = [m.pn for m in analysis.token_metrics]
        ps_scores = [m.ps for m in analysis.token_metrics]
        pns_scores = [m.pns for m in analysis.token_metrics]
        
        ax2.plot(steps, pn_scores, 'g-', label='Necessity (PN)')
        ax2.plot(steps, ps_scores, 'y-', label='Sufficiency (PS)')
        ax2.plot(steps, pns_scores, 'm-', label='Combined (PNS)')
        ax2.set_title('Causal Analysis Metrics')
        ax2.set_ylabel('Probability')
        ax2.legend()
        
        confidence_colors = {
            'High': 'green',
            'Medium': 'yellow',
            'Low': 'red'
        }
        
        confidence_values = [confidence_colors[m.confidence_region] 
                           for m in analysis.token_metrics]
        
        ax3.scatter(steps, [1]*len(steps), c=confidence_values, marker='s', s=100)
        ax3.set_title('Confidence Regions')
        ax3.set_yticks([])
        
        for point in analysis.critical_points:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=point, color='red', linestyle='--', alpha=0.3)
        
        for start, end in analysis.high_confidence_regions:
            for ax in [ax1, ax2, ax3]:
                ax.axvspan(start, end, color='green', alpha=0.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig

def analyze_ground_truth(
    input_text: str,
    max_steps: int = 100,
    save_path: Optional[str] = None
):
    """Main function to run ground truth analysis."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    analyzer = GroundTruthAnalyzer(model, tokenizer)
    analysis = analyzer.analyze_sequence(input_text, max_steps)
    
    print("\n=== Ground Truth Analysis ===")
    print(f"Overall Reliability: {analysis.overall_reliability:.3f}")
    print(f"\nGenerated Text: {analysis.text}")
    
    print("\nHigh Confidence Regions:")
    for start, end in analysis.high_confidence_regions:
        region_text = analysis.text.split()[start:end+1]
        print(f"  Tokens {start}-{end}: {' '.join(region_text)}")
    
    print("\nCritical Points Requiring Intervention:")
    for point in analysis.critical_points:
        words = analysis.text.split()
        start_idx = max(0, point-2)
        end_idx = min(len(words), point+3)
        context = words[start_idx:end_idx]
        
        prefix = "..." if start_idx > 0 else ""
        suffix = "..." if end_idx < len(words) else ""
        
        print(f"  Token {point}: {prefix}{' '.join(context)}{suffix}")
    
    fig = analyzer.visualize_analysis(analysis, save_path)
    
    return analysis, fig

if __name__ == "__main__":
    input_text = "The capital of France"
    analysis, fig = analyze_ground_truth(
        input_text,
        max_steps=50,
        save_path="ground_truth_analysis.pdf"
    )
    plt.show()