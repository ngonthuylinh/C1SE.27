#!/usr/bin/env python3
"""
PDF Generator for Survey Forms
Generates professional PDF forms from survey data
"""

import sys
import json
import os
from datetime import datetime
from pathlib import Path

# Install required packages if not available
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
except ImportError:
    print("Installing required packages...")
    os.system("pip install reportlab")
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

def create_custom_styles():
    """Create custom styles for the PDF"""
    styles = getSampleStyleSheet()
    
    # Title style
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    
    # Subtitle style
    styles.add(ParagraphStyle(
        name='CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.grey,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    ))
    
    # Question style
    styles.add(ParagraphStyle(
        name='Question',
        parent=styles['Normal'],
        fontSize=11,
        fontName='Helvetica-Bold',
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#333333')
    ))
    
    # Answer line style
    styles.add(ParagraphStyle(
        name='AnswerLine',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=15,
        textColor=colors.grey
    ))
    
    # Info style
    styles.add(ParagraphStyle(
        name='Info',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    ))
    
    return styles

def generate_answer_space(question_type, options=None):
    """Generate appropriate answer space based on question type"""
    if question_type == 'multiple_choice' and options:
        # Radio button style options
        answer_elements = []
        for option in options:
            answer_elements.append(f"⚪ {option}")
        return "<br/>".join(answer_elements)
    elif question_type == 'checkbox' and options:
        # Checkbox style options
        answer_elements = []
        for option in options:
            answer_elements.append(f"☐ {option}")
        return "<br/>".join(answer_elements)
    elif question_type == 'rating':
        # Rating scale 1-5
        return "1 ⚪  2 ⚪  3 ⚪  4 ⚪  5 ⚪"
    elif question_type == 'yes_no':
        # Yes/No options
        return "⚪ Yes&nbsp;&nbsp;&nbsp;&nbsp;⚪ No"
    else:
        # Default text input
        return "_" * 60

def generate_survey_pdf(form_data, output_path):
    """Generate PDF from survey form data"""
    try:
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Get custom styles
        styles = create_custom_styles()
        
        # Build document elements
        elements = []
        
        # Header
        elements.append(Paragraph(f"{form_data['title']}", styles['CustomTitle']))
        
        if form_data.get('description'):
            elements.append(Paragraph(f"{form_data['description']}", styles['CustomSubtitle']))
        
        # Survey info
        info_data = [
            ['Survey ID:', form_data['id']],
            ['Category:', form_data.get('category', 'General').title()],
            ['Created:', datetime.fromisoformat(form_data['createdAt'].replace('Z', '+00:00')).strftime('%B %d, %Y')],
            ['Questions:', str(len(form_data['questions']))]
        ]
        
        info_table = Table(info_data, colWidths=[3*cm, 6*cm])
        info_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONT', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        
        # Instructions
        instructions = """
        <b>Instructions:</b><br/>
        Please answer all questions honestly and completely. 
        Use the spaces provided below each question for your responses.
        """
        elements.append(Paragraph(instructions, styles['Info']))
        elements.append(Spacer(1, 20))
        
        # Questions
        for i, question_data in enumerate(form_data['questions'], 1):
            # Question number and text
            question_text = f"{i}. {question_data['question']}"
            if question_data.get('required'):
                question_text += " <font color='red'>*</font>"
            
            elements.append(Paragraph(question_text, styles['Question']))
            
            # Answer space
            answer_space = generate_answer_space(
                question_data.get('type', 'text'),
                question_data.get('options')
            )
            elements.append(Paragraph(answer_space, styles['AnswerLine']))
            
            # Add extra space for long answers
            if question_data.get('type') in ['text', 'textarea']:
                elements.append(Spacer(1, 20))
        
        # Footer
        elements.append(Spacer(1, 30))
        footer_text = f"""
        <b>Thank you for your participation!</b><br/>
        Generated by Form Agent AI on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
        Survey ID: {form_data['id']}
        """
        elements.append(Paragraph(footer_text, styles['Info']))
        
        # Build PDF
        doc.build(elements)
        
        print(f"PDF generated successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

def main():
    """Main function for command line usage"""
    if len(sys.argv) != 3:
        print("Usage: python pdf_generator.py '<json_data>' '<output_path>'")
        sys.exit(1)
    
    try:
        # Parse arguments
        json_data = sys.argv[1]
        output_path = sys.argv[2]
        
        # Parse form data
        form_data = json.loads(json_data)
        
        # Generate PDF
        success = generate_survey_pdf(form_data, output_path)
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()