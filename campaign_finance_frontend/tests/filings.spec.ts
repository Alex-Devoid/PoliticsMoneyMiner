import { test, expect } from '@playwright/test'
import { setActivePinia, createPinia } from 'pinia'
import { useFilingsStore } from '../store/filings.js'   // <-- fix the path

test.beforeEach(() => {
  setActivePinia(createPinia())
})

test('increments currentId on selectNext', () => {
  const s = useFilingsStore()
  s.rows = [{ id: 'a' }, { id: 'b' }]
  s.currentId = 'a'
  s.selectNext()
  expect(s.currentId).toBe('b')
})
