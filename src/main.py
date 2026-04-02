# main.py
from caption_retrieval import extract_conditions
from inversion import ddim_inversion
from id_generation import generate_new_id
from diffusion_generation import image_generation_loop

def main():
    # 1. Inputs
    input_image = "obama_source.jpg"
    list_of_captions = [
        "A photo of a man in a suit", 
        "A photo of a smiling politician", 
        "A photo of a person indoors"
    ]
    
    # 2. Attribute/Caption Retrieval
    print("Starting Attribute Retrieval...")
    c1, c2 = extract_conditions(input_image, list_of_captions)
    
    # 3. DDIM Inversion
    print("Starting DDIM Inversion...")
    # Passing C1 as the optional condition as indicated by the diagram arrow
    noise = ddim_inversion(input_image, optional_condition_c1=c1)
    
    # 4. ID Generation
    print("Starting ID Generation...")
    new_id = generate_new_id(c2)
    
    # 5. Image Generation (Dual-path Diffusion)
    print("Starting Custom Diffusion Generation...")
    final_image = image_generation_loop(
        starting_noise=noise, 
        c1_condition=c1, 
        new_id_condition=new_id
    )
    
    print(f"Pipeline finished. Output saved to: {final_image}")

if __name__ == "__main__":
    main()
