from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import magenta, pink, blue, green


def create_simple_form():
    c = canvas.Canvas('immblank2.pdf')

    c.setFont("Courier", 20)
    c.drawCentredString(300, 700, 'Employment Form')
    c.setFont("Courier", 14)
    form = c.acroForm

    c.drawString(10, 650, 'First Name:')
    form.textfield(name='fname', tooltip='First Name',
                   x=110, y=635, borderStyle='solid',
                   borderColor=magenta, fillColor=pink,
                   width=300, fontSize=3,
                   textColor=blue, forceBorder=True)

    c.drawString(10, 600, 'Last Name:')
    form.textfield(name='lname', tooltip='Last Name',
                   x=110, y=585, borderStyle='solid',
                   borderColor=green, fillColor=magenta,
                   width=300,
                   textColor=blue, forceBorder=True)

    c.drawString(10, 550, 'Address:')
    form.textfield(name='address', tooltip='Address',
                   x=110, y=535, borderStyle='inset',
                   width=400, forceBorder=True)

    c.drawString(10, 500, 'City:')
    form.textfield(name='city', tooltip='City',
                   x=110, y=485, borderStyle='inset',
                   forceBorder=True)

    c.drawString(250, 500, 'State:')
    form.textfield(name='state', tooltip='State',
                   x=350, y=485, borderStyle='inset',
                   forceBorder=True)

    c.drawString(10, 450, 'Zip Code:')
    form.choice(name='zip_code', tooltip='Zip Code',
                   x=110, y=435, borderStyle='solid', options=['hello', 'there', 'my', 'boi', 'edit'],
                   borderWidth=0, height=10, fontSize=8, value='hello',
                   fieldFlags=1<<18, forceBorder=True)

    c.save()


if __name__ == '__main__':
    create_simple_form()