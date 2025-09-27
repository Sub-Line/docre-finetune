"""
Preprocessing pipeline for RE-DocRED data.
Converts relation codes to human-readable labels and prepares data for training.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from relation codes to human-readable labels
# This is based on Wikidata property codes used in DocRED/RE-DocRED
RELATION_CODE_TO_LABEL = {
    "P1": "instance_of",
    "P6": "head_of_government",
    "P17": "country",
    "P18": "place_of_birth",
    "P19": "place_of_birth",
    "P20": "place_of_death",
    "P22": "father",
    "P25": "mother",
    "P26": "spouse",
    "P27": "country_of_citizenship",
    "P30": "continent",
    "P31": "instance_of",
    "P35": "head_of_state",
    "P36": "capital",
    "P37": "official_language",
    "P39": "position_held",
    "P40": "child",
    "P50": "author",
    "P54": "member_of_sports_team",
    "P57": "director",
    "P58": "screenwriter",
    "P69": "educated_at",
    "P86": "composer",
    "P102": "member_of_political_party",
    "P108": "employer",
    "P112": "founded_by",
    "P118": "league",
    "P123": "publisher",
    "P127": "owned_by",
    "P131": "located_in_administrative_division",
    "P136": "genre",
    "P137": "operator",
    "P140": "religion",
    "P150": "contains_administrative_division",
    "P155": "follows",
    "P156": "followed_by",
    "P159": "headquarters_location",
    "P161": "cast_member",
    "P162": "producer",
    "P166": "award_received",
    "P170": "creator",
    "P171": "parent_taxon",
    "P172": "ethnic_group",
    "P175": "performer",
    "P176": "manufacturer",
    "P178": "developer",
    "P179": "part_of_the_series",
    "P190": "sister_city",
    "P194": "legislative_body",
    "P206": "located_next_to_body_of_water",
    "P241": "military_branch",
    "P264": "record_label",
    "P272": "production_company",
    "P276": "location",
    "P279": "subclass_of",
    "P281": "postal_code",
    "P282": "writing_system",
    "P344": "director_of_photography",
    "P355": "subsidiary",
    "P361": "part_of",
    "P364": "original_language_of_work",
    "P400": "platform",
    "P403": "mouth_of_watercourse",
    "P449": "original_network",
    "P463": "member_of",
    "P488": "chairperson",
    "P495": "country_of_origin",
    "P527": "has_part",
    "P551": "residence",
    "P569": "date_of_birth",
    "P570": "date_of_death",
    "P571": "inception",
    "P576": "dissolved_or_abolished",
    "P577": "publication_date",
    "P580": "start_time",
    "P582": "end_time",
    "P585": "point_in_time",
    "P607": "conflict",
    "P674": "characters",
    "P676": "lyrics_by",
    "P706": "located_on_terrain_feature",
    "P710": "participant",
    "P737": "influenced_by",
    "P740": "location_of_formation",
    "P749": "parent_organization",
    "P800": "notable_work",
    "P807": "separated_from",
    "P840": "narrative_location",
    "P937": "work_location",
    "P1001": "applies_to_jurisdiction",
    "P1056": "product_or_material_produced",
    "P1198": "unemployment_rate",
    "P1336": "territory_claimed_by",
    "P1344": "participant_of",
    "P1365": "replaces",
    "P1366": "replaced_by",
    "P1376": "capital_of",
    "P1412": "languages_spoken_or_signed",
    "P1441": "present_in_work",
    "P3373": "sibling"
}


def load_relation_mapping(rel_info_path: str) -> Dict[str, str]:
    """Load relation mapping from rel_info.json if available."""
    if not Path(rel_info_path).exists():
        logger.warning(f"Relation info file not found at {rel_info_path}. Using default mapping.")
        return RELATION_CODE_TO_LABEL

    try:
        with open(rel_info_path, 'r') as f:
            rel_info = json.load(f)

        # Extract mapping from rel_info structure
        mapping = {}
        for rel_id, info in rel_info.items():
            if isinstance(info, dict) and 'name' in info:
                # Convert name to snake_case label
                label = info['name'].lower().replace(' ', '_').replace('-', '_')
                mapping[rel_id] = label
            else:
                # Fallback to default if available
                mapping[rel_id] = RELATION_CODE_TO_LABEL.get(rel_id, f"relation_{rel_id}")

        logger.info(f"Loaded {len(mapping)} relation mappings from {rel_info_path}")
        return mapping

    except Exception as e:
        logger.error(f"Error loading relation info: {e}. Using default mapping.")
        return RELATION_CODE_TO_LABEL


def preprocess_document(doc: Dict[str, Any], relation_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Preprocess a single RE-DocRED document."""
    # Extract text from sentences
    sentences = doc.get('sents', [])
    full_text = ' '.join([' '.join(sent) for sent in sentences])

    # Process entities
    entities = []
    for entity in doc.get('vertexSet', []):
        # Each entity can have multiple mentions
        entity_info = {
            'mentions': entity,
            'name': entity[0]['name'] if entity and 'name' in entity[0] else '',
            'type': entity[0]['type'] if entity and 'type' in entity[0] else 'ENTITY'
        }
        entities.append(entity_info)

    # Process relations/labels
    relations = []
    for label in doc.get('labels', []):
        relation_code = label.get('r', '')
        relation_label = relation_mapping.get(relation_code, f"unknown_relation_{relation_code}")

        relation_info = {
            'head': label.get('h', -1),
            'tail': label.get('t', -1),
            'relation_code': relation_code,
            'relation_label': relation_label,
            'evidence': label.get('evidence', [])
        }
        relations.append(relation_info)

    return {
        'title': doc.get('title', ''),
        'text': full_text,
        'sentences': sentences,
        'entities': entities,
        'relations': relations,
        'doc_id': doc.get('title', '').replace(' ', '_').lower()
    }


def create_training_examples(processed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create training examples from a processed document."""
    examples = []

    for relation in processed_doc['relations']:
        head_idx = relation['head']
        tail_idx = relation['tail']

        if head_idx >= len(processed_doc['entities']) or tail_idx >= len(processed_doc['entities']):
            continue

        head_entity = processed_doc['entities'][head_idx]
        tail_entity = processed_doc['entities'][tail_idx]

        example = {
            'text': processed_doc['text'],
            'head_entity': head_entity['name'],
            'tail_entity': tail_entity['name'],
            'relation': relation['relation_label'],
            'head_type': head_entity['type'],
            'tail_type': tail_entity['type'],
            'doc_id': processed_doc['doc_id']
        }
        examples.append(example)

    return examples


def preprocess_dataset(input_file: str, output_file: str, rel_info_file: str = None):
    """Preprocess entire RE-DocRED dataset."""
    logger.info(f"Preprocessing {input_file} -> {output_file}")

    # Load relation mapping
    if rel_info_file and Path(rel_info_file).exists():
        relation_mapping = load_relation_mapping(rel_info_file)
    else:
        relation_mapping = RELATION_CODE_TO_LABEL

    # Load input data
    with open(input_file, 'r') as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} documents")

    # Process documents
    all_examples = []
    processed_docs = []

    for doc in raw_data:
        try:
            processed_doc = preprocess_document(doc, relation_mapping)
            processed_docs.append(processed_doc)

            # Create training examples
            examples = create_training_examples(processed_doc)
            all_examples.extend(examples)

        except Exception as e:
            logger.error(f"Error processing document {doc.get('title', 'unknown')}: {e}")
            continue

    logger.info(f"Created {len(all_examples)} training examples from {len(processed_docs)} documents")

    # Save processed data
    output_data = {
        'examples': all_examples,
        'processed_docs': processed_docs,
        'relation_mapping': relation_mapping,
        'stats': {
            'num_documents': len(processed_docs),
            'num_examples': len(all_examples),
            'num_relations': len(set(ex['relation'] for ex in all_examples))
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved preprocessed data to {output_file}")
    return output_data


if __name__ == "__main__":
    # Example usage
    data_dir = "./data"

    files_to_process = [
        ("train_annotated.json", "train_processed.json"),
        ("dev.json", "dev_processed.json"),
        ("test.json", "test_processed.json")
    ]

    rel_info_file = Path(data_dir) / "rel_info.json"

    for input_file, output_file in files_to_process:
        input_path = Path(data_dir) / input_file
        output_path = Path(data_dir) / output_file

        if input_path.exists():
            preprocess_dataset(str(input_path), str(output_path), str(rel_info_file))
        else:
            logger.warning(f"Input file not found: {input_path}")