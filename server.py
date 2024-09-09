from flask import Flask, request, jsonify
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from astrapy import DataAPIClient
import os
from dotenv import load_dotenv
from flask_cors import CORS  # Add this

# import {OpenAIStream, StreamingTextResponse} from "ai";

load_dotenv()

genai.configure(api_key=os.environ.get("AI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)

# Enable CORS
# Enable CORS and handle preflight requests
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Environment variables
db_token = os.environ.get('DB_TOKEN')
db_endpoint = os.environ.get('DB_ENDPOINT')

# Initialize the DB client
client = DataAPIClient(db_token)
db = client.get_database_by_api_endpoint(
  db_endpoint
)
print(f"Connected to Astra DB: {db.info().name}")
print(f"Connected to Astra DB: {db.list_collection_names()}")

# Encoder
encoder = SentenceTransformer("all-mpnet-base-v2")

@app.route('/')
def home():
    return "Healthy Server!"

@app.route('/api/chat', methods=['POST'])
def chat_handler():
    if request.method == 'OPTIONS':
        # Handle preflight requests
        response = jsonify({'status': 'preflight'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response
    
    try:
        # Parse the JSON from the request
        data = request.get_json()
        messages = data.get('messages', [])
        print(messages)

        # Get the latest message content
        latest_message = messages[-1].get('content')
        print(latest_message)
        docContext = ""

        # create embedding
        search_vector=encoder.encode(latest_message)
        print('search_vector')

        collection = db.get_collection("portfolio");

        results = collection.find(
            sort={"$vector": search_vector},
            limit=2,
        )
        print("Vector search results:")
        doc_context = (
        "\nSTART CONTEXT\n"
        + "\n".join(doc['title']+": "+doc['description'] for doc in results)
        + "\nEND CONTEXT"
        )

        print('doc_context')
        print(doc_context)

        # Create the RAG prompt
        rag_prompt = (
            "You are an AI assistant answering questions as Sohail in his Portfolio Web App. Talk like a human."+
            "Format responses using markdown where applicable.\n"+
            "Just give the required answers and to the point. If the answer is not provided in the context, the AI assistant will say, "+
            "\"I'm sorry, I do not know the answer\"."+
            "The context is provided below"+
            f"{doc_context}\n"+
            "The Question is below\n"+
            f"{latest_message}\n"
        )

        print(rag_prompt)  # To check the output

        response = model.generate_content(
        rag_prompt
        )

        result=response.text;
        print(result)

        return jsonify({
            'msg': result
        }), 200

        # const stream = OpenAIStream(response);
        # return new StreamingTextResponse(stream);

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000, debug=True)

#     const documents = await cursor.toArray();

#     docContext = `
#           START CONTEXT
#           ${documents?.map((doc) => doc.description).join("\n")}
#           END CONTEXT
#           `;

#     const ragPrompt = [
#       {
#         role: "system",
#         content: `
#               You are an AI assistant answering questions as Piyush Agarwal in his Portfolio App. 
#               Format responses using markdown where applicable.
#               ${docContext}
#               If the answer is not provided in the context, the AI assistant will say, 
#               "I'am sorry, I do not know the answer".
#               `,
#       },
#     ];

#     const response = await openai.chat.completions.create({
#       model: "gpt-3.5-turbo",
#       stream: true,
#       messages: [...ragPrompt, ...messages],
#     });
#     const stream = OpenAIStream(response);
#     return new StreamingTextResponse(stream);
#   } catch (e) {
#     throw e;
#   }
# }