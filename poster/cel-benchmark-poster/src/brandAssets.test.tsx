import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import brands from '../../research/brand-assets.json'
import { BrandStrip } from './components/poster/BrandStrip'

describe('reference poster logos', () => {
  it('preserves all three supplied logo files byte for byte', () => {
    const { container } = render(<BrandStrip />)
    expect(brands.assets.map((asset) => asset.id)).toEqual(['pwr', 'genwro', 'tooploox'])
    expect(container.querySelectorAll('img')).toHaveLength(3)
    for (const brand of brands.assets) {
      const bytes = readFileSync(path.resolve(process.cwd(), '../..', brand.localFile))
      expect(createHash('sha256').update(bytes).digest('hex')).toBe(brand.sha256)
      expect(container.querySelector(`[data-brand-id="${brand.id}"]`)?.getAttribute('alt')).toBe(brand.label)
    }
  })
})
