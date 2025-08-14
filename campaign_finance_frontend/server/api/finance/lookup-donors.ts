import { defineEventHandler, getQuery } from 'h3'
import { callFastApi } from '../_utils'

export default defineEventHandler(event =>
  callFastApi(event, '/finance/lookup/donors', getQuery(event))
)
