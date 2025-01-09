import re
import json
import pdfplumber  # You'll need to install this: pip install pdfplumber

def extract_qa_from_pdf(pdf_path, output_json_path):
    """
    Extract Q&A pairs from a PDF file and save them to a JSON file.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_json_path (str): Path where the JSON file will be saved
    """
    # Dictionary to store all Q&A pairs
    qa_dict = {}
    
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Extract text from all pages
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    
    # Regular expression pattern to match Q&A pairs
    # This pattern looks for "Q.number" followed by text and "A.number" followed by text
    pattern = r'Q\.(\d+)\.(.*?)(?=Q\.\d+\.|A\.\d+\.)(A\.\d+\.)(.+?)(?=Q\.\d+\.|$)'
    
    # Find all matches in the text
    matches = re.finditer(pattern, text, re.DOTALL)
    
    # Process each match
    for match in matches:
        q_number = match.group(1)
        question = match.group(2).strip()
        answer = match.group(4).strip()
        
        # Create a dictionary for this Q&A pair
        qa_dict[f"Q{q_number}"] = {
            "question": question,
            "answer": answer
        }
    
    # Sort the dictionary by question number
    sorted_qa = dict(sorted(qa_dict.items(), 
                          key=lambda x: int(x[0].replace('Q', ''))))
    
    # Write to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_qa, f, indent=4, ensure_ascii=False)
    
    return sorted_qa

def main():
    # Example usage
    pdf_path = '08.30.23.round1questionsanswers.pdf'  # Replace with your PDF file path
    output_json_path = 'qa_output2023.json'  # Replace with desired output path
    
    try:
        qa_pairs = extract_qa_from_pdf(pdf_path, output_json_path)
        print(f"Successfully extracted {len(qa_pairs)} Q&A pairs")
        print(f"JSON file has been saved to {output_json_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
