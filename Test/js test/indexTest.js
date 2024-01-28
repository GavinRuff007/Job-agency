
import { quiz } from './quiz';

describe('quiz', () => {
  // Tests whether the `totalQuestions` property returns the correct number of questions.
  it('should return the correct number of questions', () => {
    const totalQuestions = quiz.totalQuestions;

    expect(totalQuestions).toBe(5);
  });

  // Tests whether the `questions` property returns the correct question for the given index.
  it('should return the correct question', () => {
    const question = quiz.questions[0];

    expect(question).toEqual({
      id: 1,
      question: 'Agile Manifesto defines what principles?',
      answers: [
        '(a) Commitment to the customer, change-oriented, collaboration, and respect for individuals',
        '(b) Principles of privacy, advertising, transparency, and diversity',
        '(c) Principles of health, environmental, quality of life, and social justice',
        '(d) Principles of facilitating communication, teamwork, leadership, and creativity'
      ],
      correctAnswer: '(a) Commitment to the customer, change-oriented, collaboration, and respect for individuals'
    });
  });
});

// Tests whether the questions are sorted correctly
it('should sort questions correctly', () => {
  const quiz = {
    questions: [
      { id: 1, question: 'Agile Manifesto defines what principles?' },
      { id: 2, question: 'How does Agile help improve communication in a team?' },
      { id: 3, question: 'What are the pros and cons of using Agile?' },
      { id: 4, question: 'How can the Agile experience be improved in different projects?' }
    ]
  };

  expect(quiz.questions).toEqual([
    { id: 1, question: 'Agile Manifesto defines what principles?' },
    { id: 2, question: 'How does Agile help improve communication in a team?' },
    { id: 3, question: 'What are the pros and cons of using Agile?' },
    { id: 4, question: 'How can the Agile experience be improved in different projects?' }
  ]);
});

// Tests whether the questions have the correct format
it('should have questions with the correct format', () => {
  const quiz = {
    questions: [
      { id: 1, question: 'Agile Manifesto defines what principles?' },
      { id: 2, question: 'How does Agile help improve communication in a team?' },
      { id: 3, question: 'What are the pros and cons of using Agile?' },
      { id: 4, question: 'How can the Agile experience be improved in different projects?' }
    ]
  };

  for (const question of quiz.questions) {
    expect(question.id).toBeInstanceOf(Number);
    expect(question.question).toBeInstanceOf(String);
    expect(question.answers).toBeInstanceOf(Array);
    expect(question.correctAnswer).toBeInstanceOf(String);
  }
});

// Tests whether the correct answer is correct for each question
it('should have the correct correct answer for each question', () => {
  const quiz = {
    questions: [
      { id: 1, question: 'Agile Manifesto defines what principles?', correctAnswer: '(a) Commitment to the customer, change-oriented, collaboration, and respect for individuals' },
      { id: 2, question: 'How does Agile help improve communication in a team?', correctAnswer: '(b) By creating more interaction and collaboration' },
      { id: 3, question: 'What are the pros and cons of using Agile?', correctAnswer: '(a) Increased team collaboration / increased team tension' },
      { id: 4, question: 'How can the Agile experience be improved in different projects?', correctAnswer: '(b) By creating a suitable culture and continuous education' }
    ]
  };

  for (const question of quiz.questions) {
    expect(question.correctAnswer).toEqual(quiz.questions[question.id - 1].correctAnswer);
  }
});
