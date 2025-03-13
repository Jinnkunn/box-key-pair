import os
import argparse
import pandas as pd

def create_output_directories():
    """Create necessary output directories if they don't exist"""
    # Base directories
    base_dirs = ['data', 'visualization', 'reports', 'output']
    
    # Individual model directories
    individual_dirs = [
        'output/individual/data',
        'output/individual/visualization',
        'output/individual/reports',
        'output/individual/visualization/parameter',
        'output/individual/visualization/learning',
        'output/individual/visualization/strategy',
        'output/individual/visualization/performance',
        'output/individual/visualization/cluster',
        'output/individual/reports/parameter',
        'output/individual/reports/learning',
        'output/individual/reports/strategy',
        'output/individual/reports/performance',
        'output/individual/reports/cluster',
    ]
    
    # Generative model directories
    generative_dirs = [
        'output/generative/data',
        'output/generative/visualization',
        'output/generative/visualization/parameter',
        'output/generative/visualization/posterior',
        'output/generative/visualization/recovery',
        'output/generative/visualization/selection',
        'output/generative/visualization/validation',
        'output/generative/visualization/uncertainty',
        'output/generative/visualization/detailed',
        'output/generative/reports',
        'output/generative/reports/parameter',
        'output/generative/reports/posterior',
        'output/generative/reports/recovery',
        'output/generative/reports/selection',
        'output/generative/reports/validation',
        'output/generative/reports/uncertainty',
        'output/generative/reports/detailed',
    ]
    
    # Create all directories
    for directory in base_dirs + individual_dirs + generative_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def load_data() -> pd.DataFrame:
    """Load and merge data from original files"""
    print("\nLoading data files...")
    
    # Load data from original files
    long_data = pd.read_csv('data/dollhouse_long.csv')
    short_data = pd.read_csv('data/dollhouse_short.csv')
    
    # Add gender and age information from short_data to long_data
    gender_age_dict = dict(zip(short_data['ID'], zip(short_data['Gender'], short_data['Age'])))
    
    # Update long_data with gender and age information
    for idx, row in long_data.iterrows():
        if row['ID'] in gender_age_dict:
            gender, age = gender_age_dict[row['ID']]
            long_data.at[idx, 'Gender'] = gender
            long_data.at[idx, 'Age'] = age
    
    # Merge data
    merged_data = pd.merge(
        long_data,
        short_data[['ID', 'NumUnlock', 'Solved', 'UnlockTime']],
        on='ID',
        how='left'
    )
    
    print(f"Loaded {len(merged_data)} trials from {len(merged_data['ID'].unique())} participants")
    return merged_data

def main():
    """Main entry point for the analysis pipeline"""
    parser = argparse.ArgumentParser(description='Run analysis pipeline for box key-pair task')
    
    # Add arguments
    parser.add_argument('--plots-only', action='store_true',
                      help='Only generate plots from saved data')
    parser.add_argument('--model', type=str, choices=['generative', 'individual', 'both'],
                      default='both', help='Which model to run (default: both)')
    
    args = parser.parse_args()
    
    # Create output directories
    print("\nCreating output directories...")
    create_output_directories()
    
    # Load data first
    data = None if args.plots_only else load_data()
    
    # Run analysis pipeline based on arguments
    if args.plots_only:
        print("\nGenerating plots from saved data...")
        from individual_model import analyze_individual_differences
        analyze_individual_differences('output/individual', generate_plots=True)
    else:
        if args.model in ['generative', 'both']:
            print("\nStep 1: Running Generative Model")
            from generative_model import GenerativeModel
            model = GenerativeModel()
            model.run_generative_model(data=data, output_dir='output/generative')
        
        if args.model in ['individual', 'both']:
            print("\nStep 2: Running Individual Differences Analysis")
            from individual_model import analyze_individual_differences
            analyze_individual_differences(data=data, output_dir='output/individual', generate_plots=True)
        
        print("\nAnalysis pipeline completed successfully!")

if __name__ == '__main__':
    main()