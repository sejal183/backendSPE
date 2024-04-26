from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.pipelines import TokenClassificationPipeline
from transformers.pipelines import AggregationStrategy 
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/extract_keyphrases": {"origins": "*"}})


# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

# Load pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)

@app.route('/extract_keyphrases', methods=['POST'])
def extract_keyphrases():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if 'text' in data:
                text = data['text']
                keyphrases = extractor(text)
                # Convert ndarray to list
                keyphrases_list = keyphrases.tolist()
                return jsonify({"keyphrases": keyphrases_list})
            else:
                return jsonify({"error": "Text input is missing"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    app.run(debug=True)

