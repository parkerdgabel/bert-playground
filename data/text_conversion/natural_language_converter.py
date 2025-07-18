"""
Natural language text converter with context-aware generation.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import random
from loguru import logger

from .base_converter import BaseTextConverter, TextConversionConfig


@dataclass
class NLConfig(TextConversionConfig):
    """Configuration for natural language conversion."""
    
    # Sentence templates
    intro_templates: List[str] = field(default_factory=lambda: [
        "This sample has the following characteristics:",
        "The data shows:",
        "We observe the following:",
        "The record indicates:",
        "Analysis reveals:",
    ])
    
    feature_templates: Dict[str, List[str]] = field(default_factory=dict)
    
    outro_templates: List[str] = field(default_factory=lambda: [
        "",
        "This completes the sample description.",
        "These are the key features.",
        "End of record.",
    ])
    
    # Sentence construction
    use_pronouns: bool = True
    pronoun_mapping: Dict[str, str] = field(default_factory=lambda: {
        "male": "he",
        "female": "she",
        "other": "they",
    })
    
    # Context awareness
    context_fields: List[str] = field(default_factory=list)
    context_rules: Dict[str, Callable[[Dict[str, Any]], str]] = field(default_factory=dict)
    
    # Style options
    style: str = "descriptive"  # "descriptive", "narrative", "technical", "casual"
    vary_sentence_structure: bool = True
    use_connectives: bool = True
    connectives: List[str] = field(default_factory=lambda: [
        "Additionally,", "Furthermore,", "Moreover,", "Also,", "In addition,",
    ])
    
    # Feature grouping
    feature_priority: Dict[str, int] = field(default_factory=dict)
    group_related_features: bool = True
    
    # Comparison and relations
    enable_comparisons: bool = False
    comparison_baselines: Dict[str, Any] = field(default_factory=dict)


class NaturalLanguageConverter(BaseTextConverter):
    """Convert data to natural language with context awareness."""
    
    def __init__(self, config: Optional[NLConfig] = None):
        """
        Initialize natural language converter.
        
        Args:
            config: Natural language configuration
        """
        if config is None:
            config = NLConfig()
        super().__init__(config)
        
        self.config: NLConfig = config
        self._style_handlers = {
            "descriptive": self._descriptive_style,
            "narrative": self._narrative_style,
            "technical": self._technical_style,
            "casual": self._casual_style,
        }
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to natural language text."""
        # Get style handler
        style_handler = self._style_handlers.get(
            self.config.style,
            self._descriptive_style
        )
        
        # Generate text using selected style
        return style_handler(data)
    
    def _descriptive_style(self, data: Dict[str, Any]) -> str:
        """Generate descriptive style text."""
        sentences = []
        
        # Add introduction
        intro = self._select_intro(data)
        if intro:
            sentences.append(intro)
        
        # Get and order features
        features = self._get_ordered_features(data)
        
        # Generate feature sentences
        feature_sentences = self._generate_feature_sentences(data, features)
        
        # Add connectives if enabled
        if self.config.use_connectives and len(feature_sentences) > 1:
            feature_sentences = self._add_connectives(feature_sentences)
        
        sentences.extend(feature_sentences)
        
        # Add comparisons if enabled
        if self.config.enable_comparisons:
            comparison_sentences = self._generate_comparisons(data)
            sentences.extend(comparison_sentences)
        
        # Add outro
        outro = self._select_outro(data)
        if outro:
            sentences.append(outro)
        
        return " ".join(sentences)
    
    def _narrative_style(self, data: Dict[str, Any]) -> str:
        """Generate narrative style text."""
        # Create a story-like description
        context = self._extract_context(data)
        
        sentences = []
        
        # Opening with context
        if context.get("subject"):
            sentences.append(f"{context['subject']} is characterized by several notable features.")
        else:
            sentences.append("This case presents an interesting profile.")
        
        # Weave features into narrative
        features = self._get_ordered_features(data)
        grouped = self._group_related_features(features, data)
        
        for group in grouped:
            group_sentence = self._create_narrative_sentence(data, group, context)
            sentences.append(group_sentence)
        
        return " ".join(sentences)
    
    def _technical_style(self, data: Dict[str, Any]) -> str:
        """Generate technical style text."""
        parts = []
        
        # Header
        parts.append("SAMPLE CHARACTERISTICS:")
        
        # Features in technical format
        features = self._get_ordered_features(data)
        for feature in features:
            value = self.get_field_value(data, feature)
            if value is not None:
                formatted = self._format_technical_feature(feature, value)
                parts.append(f"- {formatted}")
        
        return "\n".join(parts)
    
    def _casual_style(self, data: Dict[str, Any]) -> str:
        """Generate casual style text."""
        sentences = []
        
        # Casual intro
        intros = [
            "Here's what we've got:",
            "Let me tell you about this one:",
            "Check this out:",
            "So here's the deal:",
        ]
        sentences.append(random.choice(intros) if self.config.augment else intros[0])
        
        # Features in casual tone
        features = self._get_ordered_features(data)
        for i, feature in enumerate(features[:3]):  # Limit to top features
            value = self.get_field_value(data, feature)
            if value is not None:
                casual_sentence = self._create_casual_sentence(feature, value, i == 0)
                sentences.append(casual_sentence)
        
        return " ".join(sentences)
    
    def _select_intro(self, data: Dict[str, Any]) -> str:
        """Select appropriate introduction."""
        if self.config.augment and len(self.config.intro_templates) > 1:
            return random.choice(self.config.intro_templates)
        elif self.config.intro_templates:
            return self.config.intro_templates[0]
        return ""
    
    def _select_outro(self, data: Dict[str, Any]) -> str:
        """Select appropriate outro."""
        if self.config.augment and len(self.config.outro_templates) > 1:
            return random.choice(self.config.outro_templates)
        elif self.config.outro_templates:
            return self.config.outro_templates[0]
        return ""
    
    def _get_ordered_features(self, data: Dict[str, Any]) -> List[str]:
        """Get features ordered by priority."""
        features = self._get_fields_to_use(data)
        
        # Sort by priority if defined
        if self.config.feature_priority:
            features.sort(
                key=lambda f: self.config.feature_priority.get(f, 999)
            )
        
        return features
    
    def _generate_feature_sentences(
        self,
        data: Dict[str, Any],
        features: List[str]
    ) -> List[str]:
        """Generate sentences for features."""
        sentences = []
        
        for feature in features:
            value = self.get_field_value(data, feature)
            if value is None and self.config.skip_missing:
                continue
            
            # Check for custom template
            if feature in self.config.feature_templates:
                templates = self.config.feature_templates[feature]
                template = random.choice(templates) if self.config.augment else templates[0]
                
                try:
                    sentence = template.format(
                        value=self._format_feature_value(feature, value),
                        feature=feature,
                        **data
                    )
                except:
                    # Fallback
                    sentence = f"The {feature} is {self._format_feature_value(feature, value)}."
            else:
                # Default sentence
                sentence = self._create_default_sentence(feature, value)
            
            sentences.append(sentence)
        
        return sentences
    
    def _create_default_sentence(self, feature: str, value: Any) -> str:
        """Create default sentence for a feature."""
        formatted_value = self._format_feature_value(feature, value)
        
        if self.config.vary_sentence_structure:
            templates = [
                f"The {feature} is {formatted_value}.",
                f"We observe a {feature} of {formatted_value}.",
                f"The data shows {feature}: {formatted_value}.",
                f"For {feature}, we have {formatted_value}.",
            ]
            return random.choice(templates) if self.config.augment else templates[0]
        else:
            return f"The {feature} is {formatted_value}."
    
    def _format_feature_value(self, feature: str, value: Any) -> str:
        """Format feature value for natural language."""
        if value is None:
            return "not specified"
        elif isinstance(value, bool):
            return "present" if value else "absent"
        else:
            return str(value)
    
    def _add_connectives(self, sentences: List[str]) -> List[str]:
        """Add connectives between sentences."""
        if len(sentences) <= 1:
            return sentences
        
        result = [sentences[0]]
        
        for i, sentence in enumerate(sentences[1:], 1):
            if i < len(sentences) - 1 and random.random() < 0.5:
                connective = random.choice(self.config.connectives)
                result.append(f"{connective} {sentence.lower()}")
            else:
                result.append(sentence)
        
        return result
    
    def _extract_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context from data."""
        context = {}
        
        # Extract from context fields
        for field in self.config.context_fields:
            if field in data:
                context[field] = data[field]
        
        # Apply context rules
        for field, rule in self.config.context_rules.items():
            try:
                context[field] = rule(data)
            except:
                pass
        
        # Determine subject/pronoun
        if self.config.use_pronouns:
            gender_field = context.get("gender") or data.get("gender") or data.get("sex")
            if gender_field:
                context["pronoun"] = self.config.pronoun_mapping.get(
                    str(gender_field).lower(),
                    "they"
                )
        
        return context
    
    def _group_related_features(
        self,
        features: List[str],
        data: Dict[str, Any]
    ) -> List[List[str]]:
        """Group related features together."""
        if not self.config.group_related_features:
            return [[f] for f in features]
        
        # Simple grouping by common prefixes
        groups = {}
        ungrouped = []
        
        for feature in features:
            # Check for common prefixes
            prefix = feature.split("_")[0] if "_" in feature else None
            
            if prefix and prefix in groups:
                groups[prefix].append(feature)
            elif prefix:
                groups[prefix] = [feature]
            else:
                ungrouped.append(feature)
        
        # Convert to list of groups
        result = list(groups.values())
        result.extend([[f] for f in ungrouped])
        
        return result
    
    def _create_narrative_sentence(
        self,
        data: Dict[str, Any],
        features: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Create narrative sentence for feature group."""
        if len(features) == 1:
            feature = features[0]
            value = self.get_field_value(data, feature)
            pronoun = context.get("pronoun", "They").capitalize()
            return f"{pronoun} had a {feature} of {value}."
        else:
            # Multiple features
            parts = []
            for feature in features:
                value = self.get_field_value(data, feature)
                parts.append(f"{feature} of {value}")
            
            return f"The record shows {', '.join(parts)}."
    
    def _format_technical_feature(self, feature: str, value: Any) -> str:
        """Format feature in technical style."""
        return f"{feature.upper()}: {value}"
    
    def _create_casual_sentence(self, feature: str, value: Any, is_first: bool) -> str:
        """Create casual style sentence."""
        if is_first:
            return f"The {feature} is {value}."
        else:
            connectives = ["Also", "Plus", "And", "Oh, and"]
            connective = random.choice(connectives) if self.config.augment else "Also"
            return f"{connective}, the {feature} is {value}."
    
    def _generate_comparisons(self, data: Dict[str, Any]) -> List[str]:
        """Generate comparison sentences."""
        if not self.config.comparison_baselines:
            return []
        
        sentences = []
        
        for feature, baseline in self.config.comparison_baselines.items():
            if feature in data:
                value = data[feature]
                if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
                    if value > baseline:
                        sentences.append(f"The {feature} is above average ({value} vs {baseline}).")
                    elif value < baseline:
                        sentences.append(f"The {feature} is below average ({value} vs {baseline}).")
        
        return sentences
    
    def __repr__(self) -> str:
        """String representation."""
        return f"NaturalLanguageConverter(style={self.config.style})"