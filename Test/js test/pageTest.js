import React from 'react';
import { render, screen } from '@testing-library/react';
import Home from './page';

jest.mock('next/link', () => ({
  Link: ({ children, href }) => children, // Mock Link to avoid routing
}));

describe('Home', () => {
  it('renders the heading and button', () => {
    render(<Home />);

    const heading = screen.getByText('اپلیکیشن Quiz');
    expect(heading).toBeInTheDocument();

    const button = screen.getByRole('button', { name: /شروع آزمون/i });
    expect(button).toBeInTheDocument();
  });

  it('renders the button with correct styling', () => {
    render(<Home />);

    const button = screen.getByRole('button', { name: /شروع آزمون/i });
    expect(button).toHaveClass('px-6 py-2 text-sm rounded shadow bg-rose-100 hover:bg-rose-200 text-rose-500 w-72');
  });
});
