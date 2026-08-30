import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import brands from '../../research/brand-assets.json'
import { BrandStrip } from './components/poster/BrandStrip'

describe('poster logos', () => {
  it('preserves all five supplied logo files byte for byte in their requested columns', () => {
    const { container } = render(<><BrandStrip side="left" /><BrandStrip side="right" /></>)
    expect(brands.assets.map((asset) => asset.id)).toEqual(['pwr', 'genwro', 'tooploox', 'ecml-pkdd', 'xkdd'])
    expect(container.querySelectorAll('img')).toHaveLength(5)
    expect(Array.from(container.querySelectorAll('[data-brand-side="left"] img')).map((image) => image.getAttribute('data-brand-id'))).toEqual(['ecml-pkdd', 'xkdd'])
    expect(Array.from(container.querySelectorAll('[data-brand-side="right"] img')).map((image) => image.getAttribute('data-brand-id'))).toEqual(['pwr', 'genwro', 'tooploox'])
    for (const brand of brands.assets) {
      const bytes = readFileSync(path.resolve(process.cwd(), '../..', brand.localFile))
      expect(createHash('sha256').update(bytes).digest('hex')).toBe(brand.sha256)
      expect(container.querySelector(`[data-brand-id="${brand.id}"]`)?.getAttribute('alt')).toBe(brand.label)
    }
  })
})
