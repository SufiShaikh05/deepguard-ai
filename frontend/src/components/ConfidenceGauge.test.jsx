import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ConfidenceGauge from './ConfidenceGauge'

describe('ConfidenceGauge', () => {
  it('formats confidence as a rounded percentage', () => {
    render(<ConfidenceGauge label="fake" confidence={0.913} probabilities={{ real: 0.087, fake: 0.913 }} />)
    const elements = screen.getAllByText('91%')
    expect(elements.length).toBeGreaterThanOrEqual(1)
    elements.forEach((el) => expect(el).toBeInTheDocument())
  })
})
