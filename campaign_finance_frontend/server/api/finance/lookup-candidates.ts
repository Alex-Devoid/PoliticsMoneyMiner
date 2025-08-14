import { defineEventHandler, getQuery } from 'h3'
import { callFastApi } from '../_utils'

export default defineEventHandler(async (event) => {
  const q = getQuery(event)
  return await callFastApi(event, '/finance/lookup/candidates', q)
})
